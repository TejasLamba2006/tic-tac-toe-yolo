from __future__ import annotations

import numpy as np

from src.vision.board_detector import BoardDetectionResult
from src.vision.board_state import BoardStateEstimator, Detection
from src.vision.stability import BoardGeometryTracker


def test_board_geometry_tracker_reuses_last_good_geometry_on_miss() -> None:
    tracker = BoardGeometryTracker()
    good_corners = np.array(
        [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
        dtype=np.float32,
    )
    miss_corners = np.array(
        [[0.0, 0.0], [200.0, 0.0], [200.0, 200.0], [0.0, 200.0]],
        dtype=np.float32,
    )

    first = BoardDetectionResult(
        found=True,
        corners=good_corners,
        score=0.95,
        method="canny",
        contour_area=10000.0,
        fallback=False,
    )
    miss = BoardDetectionResult(
        found=False,
        corners=miss_corners,
        score=0.0,
        method="fallback-frame",
        contour_area=40000.0,
        fallback=True,
    )

    tracked = tracker.update(first)
    reused = tracker.update(miss)

    assert tracked.found is True
    assert reused.found is True
    assert reused.fallback is False
    assert np.allclose(reused.corners, good_corners)


def test_board_state_estimator_requires_repeated_noise_to_flip_cell() -> None:
    estimator = BoardStateEstimator(smoothing_window=5, stability_ratio=0.6)
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    stable = [Detection(0, "empty", 0.99, (0, 0, 100, 100))]
    noisy = [Detection(1, "red_ball", 0.99, (0, 0, 100, 100))]

    assert estimator.estimate(image, stable).board[0][0] == "E"
    assert estimator.estimate(image, stable).board[0][0] == "E"
    assert estimator.estimate(image, noisy).board[0][0] == "E"
    assert estimator.estimate(image, noisy).board[0][0] == "E"
    assert estimator.estimate(image, noisy).board[0][0] == "R"
