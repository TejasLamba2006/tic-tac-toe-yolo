from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .perspective import PerspectiveTransform, build_transform, order_points


@dataclass(frozen=True)
class BoardDetectionResult:
    """Result of board localization on the source frame."""

    found: bool
    corners: np.ndarray
    score: float
    method: str
    contour_area: float
    fallback: bool = False

    def build_transform(self, board_size: int | tuple[int, int]) -> PerspectiveTransform:
        return build_transform(self.corners, board_size)


class BoardDetector:
    """Detect the outer Tic-Tac-Toe board as a quadrilateral.

    The detector uses a two-pass strategy: edge contours first, then a thresholded
    fallback when lighting makes the edge map weak.
    """

    def __init__(
        self,
        min_area_ratio: float = 0.12,
        canny_low: int = 40,
        canny_high: int = 120,
        blur_size: int = 5,
        adaptive_block_size: int = 31,
        adaptive_c: int = 5,
        contour_epsilon_ratio: float = 0.02,
    ) -> None:
        self.min_area_ratio = min_area_ratio
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        self.adaptive_block_size = adaptive_block_size if adaptive_block_size % 2 == 1 else adaptive_block_size + 1
        self.adaptive_c = adaptive_c
        self.contour_epsilon_ratio = contour_epsilon_ratio
        # Pre-allocated kernel reused every frame to avoid repeated numpy allocation.
        self._morph_kernel = np.ones((3, 3), dtype=np.uint8)

    def detect(self, frame: np.ndarray) -> BoardDetectionResult:
        height, width = frame.shape[:2]
        frame_area = float(height * width)

        candidates: list[tuple[np.ndarray, float, str]] = []
        for method, mask in self._candidate_masks(frame):
            for contour in self._contours(mask):
                candidate = self._quadrilateral_from_contour(
                    contour, frame_area, method)
                if candidate is not None:
                    candidates.append(candidate)

        if not candidates:
            fallback_corners = np.array(
                [[0.0, 0.0], [width - 1.0, 0.0],
                    [width - 1.0, height - 1.0], [0.0, height - 1.0]],
                dtype=np.float32,
            )
            corners, area, method = max(candidates, key=lambda item: item[1])

            # --- DEBUG: draw the detected quad on a copy of the frame ---
            debug_frame = frame.copy()
            pts = order_points(corners).astype(np.int32)
            cv2.polylines(debug_frame, [pts], isClosed=True, color=(
                0, 255, 0), thickness=3)
            for pt in pts:
                cv2.circle(debug_frame, tuple(pt), 8, (0, 0, 255), -1)
            cv2.imshow("5 - Detected Board Quad", debug_frame)
            cv2.waitKey(1)
            # -----------------------------------------------------------
            return BoardDetectionResult(
                found=False,
                corners=fallback_corners,
                score=0.0,
                method="fallback-frame",
                contour_area=frame_area,
                fallback=True,
            )

        corners, area, method = max(candidates, key=lambda item: item[1])
        return BoardDetectionResult(
            found=True,
            corners=order_points(corners),
            score=min(1.0, area / frame_area),
            method=method,
            contour_area=area,
            fallback=False,
        )

    def _candidate_masks(self, frame: np.ndarray):
        # gray + blur computed once; both passes reuse the same blurred image.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
        cv2.imshow("1 - Grayscale", gray)        # <-- ADD
        cv2.imshow("2 - Blurred", blurred)       # <-- ADD
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, self._morph_kernel, iterations=2)
        yield "canny", edges

        adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.adaptive_block_size,
            self.adaptive_c,
        )
        adaptive = cv2.morphologyEx(
            adaptive, cv2.MORPH_CLOSE, self._morph_kernel, iterations=2)
        cv2.imshow("4 - Adaptive Threshold", adaptive)
        cv2.waitKey(1)   # <-- needed to flush all imshow windows
        yield "adaptive", adaptive

    def _contours(self, mask: np.ndarray):
        # Modern OpenCV always returns a 2-tuple (contours, hierarchy).
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _quadrilateral_from_contour(
        self,
        contour: np.ndarray,
        frame_area: float,
        method: str,
    ) -> tuple[np.ndarray, float, str] | None:
        area = abs(cv2.contourArea(contour))
        if area < frame_area * self.min_area_ratio:
            return None

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return None

        approximation = cv2.approxPolyDP(
            contour, self.contour_epsilon_ratio * perimeter, True)
        if len(approximation) == 4 and cv2.isContourConvex(approximation):
            return approximation.reshape(4, 2).astype(np.float32), area, method

        if len(contour) >= 4:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect).astype(np.float32)
            return box, area * 0.9, f"{method}-minrect"

        return None
