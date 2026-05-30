from __future__ import annotations

import numpy as np

from src.vision.board_state import BoardStateEstimator, Detection, format_board


def test_board_state_assigns_detections_to_cells() -> None:
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    detections = [
        Detection(0, "empty", 0.99, (0, 0, 100, 100)),
        Detection(1, "red_ball", 0.98, (100, 0, 200, 100)),
        Detection(2, "yellow_ball", 0.97, (200, 0, 300, 100)),
        Detection(0, "empty", 0.99, (0, 100, 100, 200)),
        Detection(0, "empty", 0.99, (100, 100, 200, 200)),
        Detection(0, "empty", 0.99, (200, 100, 300, 200)),
        Detection(1, "red_ball", 0.96, (0, 200, 100, 300)),
        Detection(2, "yellow_ball", 0.95, (100, 200, 200, 300)),
        Detection(0, "empty", 0.99, (200, 200, 300, 300)),
    ]

    estimator = BoardStateEstimator(smoothing_window=1)
    observation = estimator.estimate(image, detections)

    assert observation.board == [
        ["E", "R", "Y"],
        ["E", "E", "E"],
        ["R", "Y", "E"],
    ]
    assert "E | R | Y" in format_board(observation.board)


def test_board_state_prefers_highest_scoring_detection_in_same_cell() -> None:
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    detections = [
        Detection(0, "empty", 0.40, (0, 0, 100, 100)),
        Detection(1, "red_ball", 0.90, (0, 0, 100, 100)),
    ]

    estimator = BoardStateEstimator(smoothing_window=1)
    observation = estimator.estimate(image, detections)

    assert observation.board[0][0] == "R"


def test_board_state_smoothing_keeps_recent_majority() -> None:
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    estimator = BoardStateEstimator(smoothing_window=3)

    first = [Detection(0, "empty", 0.99, (0, 0, 100, 100))]
    second = [Detection(1, "red_ball", 0.99, (0, 0, 100, 100))]
    third = [Detection(1, "red_ball", 0.99, (0, 0, 100, 100))]

    observation_1 = estimator.estimate(image, first)
    observation_2 = estimator.estimate(image, second)
    observation_3 = estimator.estimate(image, third)

    assert observation_1.board[0][0] == "E"
    assert observation_2.board[0][0] == "R"
    assert observation_3.board[0][0] == "R"
