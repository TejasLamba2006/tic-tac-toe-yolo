from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Detection:
    """Single detection returned by the YOLO wrapper."""

    class_id: int
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(frozen=True)
class BoardObservation:
    """Board state returned after cell assignment and temporal smoothing."""

    board: list[list[str]]
    cell_confidences: list[list[float]]
    cell_sources: list[list[str]]
    detections: tuple[Detection, ...] = field(default_factory=tuple)


def format_board(board: Sequence[Sequence[str]]) -> str:
    """Render the board as a simple ASCII grid."""

    rows = [" | ".join(row) for row in board]
    separator = "-" * 9
    return f"\n{separator}\n".join(rows)


def _clone_board(board: Sequence[Sequence[str]]) -> list[list[str]]:
    return [list(row) for row in board]


class BoardStateEstimator:
    """Convert detections into a 3x3 Tic-Tac-Toe board.

    The model already predicts one cell-state box per board cell. The estimator
    therefore assigns each detection to the nearest canonical cell center and then
    uses a small majority-vote history to smooth out single-frame jitter.
    """

    def __init__(
        self,
        label_to_symbol: dict[str, str] | None = None,
        rows: int = 3,
        cols: int = 3,
        minimum_confidence: float = 0.15,
        smoothing_window: int = 1,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.minimum_confidence = minimum_confidence
        self.label_to_symbol = label_to_symbol or {
            "empty": "E",
            "red_ball": "R",
            "yellow_ball": "Y",
        }
        self._history: deque[list[list[str]]] = deque(
            maxlen=max(1, smoothing_window))

    def estimate(self, image: np.ndarray, detections: Iterable[Detection]) -> BoardObservation:
        detection_list = list(detections)
        board, confidences, sources = self._estimate_once(
            image, detection_list)
        self._history.append(_clone_board(board))

        if len(self._history) > 1:
            board = self._majority_board(board)

        return BoardObservation(
            board=board,
            cell_confidences=confidences,
            cell_sources=sources,
            detections=tuple(detection_list),
        )

    def _estimate_once(
        self,
        image: np.ndarray,
        detections: Iterable[Detection],
    ) -> tuple[list[list[str]], list[list[float]], list[list[str]]]:
        height, width = image.shape[:2]
        cell_width = width / float(self.cols)
        cell_height = height / float(self.rows)

        board = [["E" for _ in range(self.cols)] for _ in range(self.rows)]
        confidences = [[0.0 for _ in range(self.cols)]
                       for _ in range(self.rows)]
        sources = [["" for _ in range(self.cols)] for _ in range(self.rows)]

        for detection in detections:
            symbol = self.label_to_symbol.get(detection.label)
            if symbol is None or detection.confidence < self.minimum_confidence:
                continue

            center_x, center_y = detection.center
            row = self._nearest_index(center_y, cell_height, self.rows)
            col = self._nearest_index(center_x, cell_width, self.cols)

            target_center_x = (col + 0.5) * cell_width
            target_center_y = (row + 0.5) * cell_height
            normalized_distance = (
                abs(center_x - target_center_x) / max(cell_width, 1.0)
                + abs(center_y - target_center_y) / max(cell_height, 1.0)
            ) / 2.0
            assignment_score = detection.confidence * \
                max(0.1, 1.0 - min(normalized_distance, 1.0))

            if assignment_score >= confidences[row][col]:
                board[row][col] = symbol
                confidences[row][col] = float(assignment_score)
                sources[row][col] = detection.label

        return board, confidences, sources

    def _nearest_index(self, value: float, step: float, limit: int) -> int:
        if step <= 0:
            return 0
        index = int(value / step)
        return max(0, min(limit - 1, index))

    def _majority_board(self, latest_board: list[list[str]]) -> list[list[str]]:
        history = list(self._history)
        if not history:
            return latest_board

        majority = [["E" for _ in range(self.cols)] for _ in range(self.rows)]
        for row_index in range(self.rows):
            for col_index in range(self.cols):
                candidates = [board[row_index][col_index] for board in history]
                counts = Counter(candidates)
                best_count = max(counts.values())
                best_symbols = {symbol for symbol,
                                count in counts.items() if count == best_count}
                if latest_board[row_index][col_index] in best_symbols:
                    majority[row_index][col_index] = latest_board[row_index][col_index]
                else:
                    majority[row_index][col_index] = next(
                        symbol for symbol in candidates[::-1] if symbol in best_symbols)
        return majority
