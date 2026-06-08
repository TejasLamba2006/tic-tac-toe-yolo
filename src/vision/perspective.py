from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PerspectiveTransform:
    """Perspective warp from a detected board into a canonical square board."""

    matrix: np.ndarray
    inverse_matrix: np.ndarray
    size: tuple[int, int]
    corners: np.ndarray


def order_points(points: np.ndarray) -> np.ndarray:
    """Return four points ordered as top-left, top-right, bottom-right, bottom-left."""

    ordered = np.asarray(points, dtype=np.float32)
    if ordered.shape != (4, 2):
        raise ValueError(f"Expected four 2D points, got shape {ordered.shape}")

    result = np.zeros((4, 2), dtype=np.float32)
    sums = ordered.sum(axis=1)
    diffs = np.diff(ordered, axis=1).reshape(-1)

    result[0] = ordered[np.argmin(sums)]
    result[2] = ordered[np.argmax(sums)]
    result[1] = ordered[np.argmin(diffs)]
    result[3] = ordered[np.argmax(diffs)]
    return result


def build_transform(corners: np.ndarray, board_size: int | tuple[int, int]) -> PerspectiveTransform:
    """Create a perspective transform that maps the board into a square image."""

    if isinstance(board_size, int):
        size = (board_size, board_size)
    else:
        size = board_size

    ordered = order_points(corners)
    destination = np.array(
        [
            [0.0, 0.0],
            [size[0] - 1.0, 0.0],
            [size[0] - 1.0, size[1] - 1.0],
            [0.0, size[1] - 1.0],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    inverse_matrix = cv2.getPerspectiveTransform(destination, ordered)
    return PerspectiveTransform(matrix=matrix, inverse_matrix=inverse_matrix, size=size, corners=ordered)


def warp_image(image: np.ndarray, transform: PerspectiveTransform) -> np.ndarray:
    """Warp a frame using ``transform``."""

    return cv2.warpPerspective(image, transform.matrix, transform.size, flags=cv2.INTER_LINEAR)


def warp_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a perspective matrix to a set of 2D points."""

    points_array = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(points_array, matrix)
    return warped.reshape(-1, 2)


def canonical_cell_polygon(row: int, col: int, board_size: int) -> np.ndarray:
    """Return the four corners of a single Tic-Tac-Toe cell in warped coordinates."""

    cell_width = board_size / 3.0
    cell_height = board_size / 3.0
    x0 = col * cell_width
    y0 = row * cell_height
    x1 = (col + 1) * cell_width
    y1 = (row + 1) * cell_height
    return np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        dtype=np.float32,
    )
