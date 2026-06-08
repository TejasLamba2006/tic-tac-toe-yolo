from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .board_detector import BoardDetectionResult
from .perspective import order_points


@dataclass
class BoardGeometryTracker:
    """Smooth board corners across frames and reuse the last good quad on misses."""

    smoothing_alpha: float = 0.35
    max_misses: int = 4
    similarity_threshold: float = 0.18
    _last_result: BoardDetectionResult | None = field(
        default=None, init=False, repr=False)
    _misses: int = field(default=0, init=False, repr=False)

    def update(self, result: BoardDetectionResult) -> BoardDetectionResult:
        if result.found and not result.fallback:
            ordered_corners = order_points(result.corners)

            if self._last_result is None:
                self._last_result = self._copy_result(
                    result, ordered_corners, result.method)
                self._misses = 0
                return self._last_result

            distance = self._normalized_corner_distance(
                ordered_corners, self._last_result.corners)
            if distance <= self.similarity_threshold:
                smoothed = (1.0 - self.smoothing_alpha) * \
                    self._last_result.corners + self.smoothing_alpha * ordered_corners
                self._last_result = BoardDetectionResult(
                    found=True,
                    corners=smoothed.astype(np.float32),
                    score=max(self._last_result.score, result.score),
                    method=f"{result.method}-smoothed",
                    contour_area=result.contour_area,
                    fallback=False,
                )
            else:
                self._last_result = self._copy_result(
                    result, ordered_corners, result.method)

            self._misses = 0
            return self._last_result

        if self._last_result is not None and self._misses < self.max_misses:
            self._misses += 1
            return BoardDetectionResult(
                found=True,
                corners=self._last_result.corners.copy(),
                score=max(0.05, self._last_result.score * 0.95),
                method=f"{self._last_result.method}-tracked",
                contour_area=self._last_result.contour_area,
                fallback=False,
            )

        return result

    def _copy_result(
        self,
        result: BoardDetectionResult,
        corners: np.ndarray,
        method: str,
    ) -> BoardDetectionResult:
        return BoardDetectionResult(
            found=result.found,
            corners=corners.astype(np.float32),
            score=result.score,
            method=method,
            contour_area=result.contour_area,
            fallback=result.fallback,
        )

    def _normalized_corner_distance(self, current: np.ndarray, reference: np.ndarray) -> float:
        # Avoid redundant array creation if inputs are already float32 ndarrays.
        if not isinstance(reference, np.ndarray) or reference.dtype != np.float32:
            reference = np.asarray(reference, dtype=np.float32)
        if not isinstance(current, np.ndarray) or current.dtype != np.float32:
            current = np.asarray(current, dtype=np.float32)
        diagonal = float(np.linalg.norm(reference[2] - reference[0]))
        if diagonal <= 1e-6:
            diagonal = float(max(np.ptp(reference[:, 0]), np.ptp(reference[:, 1]), 1.0))
        return float(np.linalg.norm(current - reference, axis=1).mean() / diagonal)
