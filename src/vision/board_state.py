from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from math import ceil
from typing import Iterable, Sequence
import cv2

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
    keeps a per-cell vote history so one noisy frame does not flip the board.
    """

    def __init__(
        self,
        label_to_symbol: dict[str, str] | None = None,
        rows: int = 3,
        cols: int = 3,
        minimum_confidence: float = 0.15,
        smoothing_window: int = 5,
        stability_ratio: float = 0.6,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.minimum_confidence = minimum_confidence
        self.stability_ratio = stability_ratio
        self.label_to_symbol = label_to_symbol or {
            "empty": "E",
            "red_ball": "R",
            "yellow_ball": "Y",
        }
        history_size = max(1, smoothing_window)
        self._cell_history: list[list[deque[str]]] = [
            [deque(maxlen=history_size) for _ in range(self.cols)]
            for _ in range(self.rows)
        ]
        self._stable_board: list[list[str]] = [
            ["E" for _ in range(self.cols)]
            for _ in range(self.rows)
        ]
        self._initialized = False
        # Cache for Fix 6: avoid re-running HSV segmentation every frame.
        self._cached_bbox: tuple[int, int, int, int] | None = None
        self._cached_bbox_image_shape: tuple[int, ...] | None = None

    def estimate(self, image: np.ndarray, detections: Iterable[Detection]) -> BoardObservation:
        detection_list = list(detections)
        instant_board, confidences, sources = self._estimate_once(
            image, detection_list)
        board = self._stabilize_board(instant_board)

        return BoardObservation(
            board=board,
            cell_confidences=confidences,
            cell_sources=sources,
            detections=tuple(detection_list),
        )

    def _find_board_bbox(self, image: np.ndarray):
        # Fix 6: cache the HSV-based bounding box so the expensive segmentation
        # (cvtColor, inRange, morphologyEx, findContours) is only re-run when the
        # image dimensions change, which is the best available proxy for a new
        # board geometry when no external geometry-change signal is passed in.
        current_shape = image.shape
        if self._cached_bbox is not None and self._cached_bbox_image_shape == current_shape:
            return self._cached_bbox

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([80, 40, 40])
        upper_blue = np.array([140, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            bbox = (0, 0, image.shape[1], image.shape[0])
        else:
            largest = max(contours, key=cv2.contourArea)
            bbox = cv2.boundingRect(largest)

        self._cached_bbox = bbox
        self._cached_bbox_image_shape = current_shape
        return bbox

    def _estimate_once(
        self,
        image: np.ndarray,
        detections: Iterable[Detection],
    ) -> tuple[list[list[str]], list[list[float]], list[list[str]]]:

        board_x, board_y, board_w, board_h = self._find_board_bbox(image)

        cell_width = board_w / float(self.cols)
        cell_height = board_h / float(self.rows)
        board = [["E" for _ in range(self.cols)] for _ in range(self.rows)]
        confidences = [[0.0 for _ in range(self.cols)]
                       for _ in range(self.rows)]
        sources = [["" for _ in range(self.cols)] for _ in range(self.rows)]
        priorities = [[(-1, 0.0) for _ in range(self.cols)]
                      for _ in range(self.rows)]

        for detection in detections:
            symbol = self.label_to_symbol.get(detection.label)
            if symbol is None or detection.confidence < self.minimum_confidence:
                continue

            row, col, assignment_score = self._best_cell_for_detection(
                detection.xyxy,
                cell_width,
                cell_height,
                board_x,
                board_y,
                confidence=detection.confidence,
            )
            priority = 1 if symbol != "E" else 0
            candidate = (priority, assignment_score)

            if candidate >= priorities[row][col]:
                board[row][col] = symbol
                confidences[row][col] = float(assignment_score)
                sources[row][col] = detection.label
                priorities[row][col] = candidate

        return board, confidences, sources

    def _best_cell_for_detection(
        self,
        xyxy,
        cell_width,
        cell_height,
        board_x,
        board_y,
        confidence: float = 1.0,
    ):

        x1, y1, x2, y2 = xyxy

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        local_x = center_x - board_x
        local_y = center_y - board_y

        col = int(local_x / cell_width)
        row = int(local_y / cell_height)

        row = max(0, min(self.rows - 1, row))
        col = max(0, min(self.cols - 1, col))

        # Fix 4: return the detection's actual confidence so higher-confidence
        # detections win when two boxes map to the same cell.
        return row, col, confidence

    def _stabilize_board(self, instant_board: list[list[str]]) -> list[list[str]]:
        if not self._initialized:
            self._initialized = True
            self._stable_board = _clone_board(instant_board)

        next_board = _clone_board(self._stable_board)
        required_votes = self._required_votes()
        self._update_stable_board(next_board, instant_board, required_votes)

        self._stable_board = next_board
        return _clone_board(self._stable_board)

    def _required_votes(self) -> int:
        history_size = self._cell_history[0][0].maxlen or 1
        return max(2, ceil(history_size * self.stability_ratio))

    def _update_stable_board(
        self,
        next_board: list[list[str]],
        instant_board: list[list[str]],
        required_votes: int,
    ) -> None:
        for row_index in range(self.rows):
            self._update_stable_row(
                next_board, instant_board, required_votes, row_index)

    def _update_stable_row(
        self,
        next_board: list[list[str]],
        instant_board: list[list[str]],
        required_votes: int,
        row_index: int,
    ) -> None:
        for col_index in range(self.cols):
            self._update_stable_cell(
                next_board, instant_board, required_votes, row_index, col_index)

    def _update_stable_cell(
        self,
        next_board: list[list[str]],
        instant_board: list[list[str]],
        required_votes: int,
        row_index: int,
        col_index: int,
    ) -> None:
        history = self._cell_history[row_index][col_index]
        history.append(instant_board[row_index][col_index])
        # Fix 3: compute Counter once and pass it to both helpers to avoid
        # building it twice per cell per frame.
        counts = Counter(history)
        label = self._select_history_label(history, counts)
        current_label = next_board[row_index][col_index]

        if self._should_flip_cell(counts, current_label, label, required_votes):
            next_board[row_index][col_index] = label

    def _select_history_label(self, history: Sequence[str], counts: Counter) -> str:
        max_votes = max(counts.values())
        for candidate in reversed(history):
            if counts[candidate] == max_votes:
                return candidate
        return history[-1]

    def _should_flip_cell(
        self,
        counts: Counter,
        current_label: str,
        new_label: str,
        required_votes: int,
    ) -> bool:
        current_votes = counts.get(current_label, 0)
        new_votes = counts.get(new_label, 0)

        return new_label != current_label and (
            (new_votes > current_votes and new_votes >= required_votes)
            or (current_votes <= 1 and new_votes == current_votes)
        )

    def _nearest_index(self, value: float, step: float, limit: int) -> int:
        if step <= 0:
            return 0
        index = int(value / step)
        return max(0, min(limit - 1, index))
