from __future__ import annotations

import numpy as np

from src.ai.move_selector import recommend_move
from src.main import analyze_frame
from src.vision.board_detector import BoardDetectionResult
from src.vision.board_state import BoardStateEstimator, Detection
from src.vision.camera import parse_camera_source


class FakeDetector:
    def __init__(self, detections):
        self._detections = list(detections)

    def predict(self, frame):
        return list(self._detections)


class FixedBoardDetector:
    def detect(self, frame):
        height, width = frame.shape[:2]
        corners = np.array(
            [[0.0, 0.0], [width - 1.0, 0.0],
                [width - 1.0, height - 1.0], [0.0, height - 1.0]],
            dtype=np.float32,
        )
        return BoardDetectionResult(
            found=True,
            corners=corners,
            score=1.0,
            method="test",
            contour_area=float(height * width),
            fallback=False,
        )


def test_camera_source_parser_handles_common_inputs() -> None:
    assert parse_camera_source("auto") == "auto"
    assert parse_camera_source("2") == 2
    assert parse_camera_source("/dev/video7") == "/dev/video7"


def test_pipeline_recommends_a_move_from_fake_detections() -> None:
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    detections = [
        Detection(0, "empty", 0.99, (0, 0, 100, 100)),
        Detection(1, "red_ball", 0.99, (100, 0, 200, 100)),
        Detection(2, "yellow_ball", 0.99, (200, 0, 300, 100)),
        Detection(0, "empty", 0.99, (0, 100, 100, 200)),
        Detection(1, "red_ball", 0.99, (100, 100, 200, 200)),
        Detection(0, "empty", 0.99, (200, 100, 300, 200)),
        Detection(2, "yellow_ball", 0.99, (0, 200, 100, 300)),
        Detection(0, "empty", 0.99, (100, 200, 200, 300)),
        Detection(0, "empty", 0.99, (200, 200, 300, 300)),
    ]

    detector = FakeDetector(detections)
    board_detector = FixedBoardDetector()
    board_estimator = BoardStateEstimator(smoothing_window=1)

    analysis = analyze_frame(
        frame=frame,
        detector=detector,
        board_detector=board_detector,
        board_estimator=board_estimator,
        ai_color="Y",
        board_size=300,
    )

    assert analysis.observation.board == [
        ["E", "R", "Y"],
        ["E", "R", "E"],
        ["Y", "E", "E"],
    ]
    assert analysis.decision.recommendation is not None


def test_recommend_move_reports_game_over_when_board_is_finished() -> None:
    board = [
        ["Y", "Y", "Y"],
        ["R", "R", "E"],
        ["E", "E", "E"],
    ]

    decision = recommend_move(board, ai_player="Y")
    assert decision.recommendation is None
    assert decision.winner == "Y"


def test_draw_panel_renders_game_over_overlay() -> None:
    from src.main import FrameAnalysis, _draw_panel
    from src.vision.board_detector import BoardDetectionResult
    from src.vision.board_state import BoardObservation
    from src.vision.perspective import PerspectiveTransform

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    warped_frame = np.zeros((320, 320, 3), dtype=np.uint8)
    board_result = BoardDetectionResult(
        found=True,
        corners=np.array([[0, 0], [320, 0], [320, 320], [0, 320]], dtype=np.float32),
        score=1.0,
        method="test",
        contour_area=320 * 320,
        fallback=False,
    )
    transform = PerspectiveTransform(
        matrix=np.eye(3, dtype=np.float32),
        inverse_matrix=np.eye(3, dtype=np.float32),
        size=(320, 320),
        corners=board_result.corners,
    )
    observation = BoardObservation(
        board=[["Y", "Y", "Y"], ["R", "R", "E"], ["E", "E", "E"]],
        cell_confidences=[[1.0] * 3] * 3,
        cell_sources=[["test"] * 3] * 3,
    )
    decision = recommend_move(observation.board, ai_player="Y")

    analysis = FrameAnalysis(
        frame=frame,
        warped_frame=warped_frame,
        board_result=board_result,
        transform=transform,
        observation=observation,
        decision=decision,
        inference_ms=1.0,
        model_run_ms=1.0,
        move_ms=1.0,
        analysis_ms=1.0,
        fps=30.0,
        detections=(),
    )

    # Render with game_over = True
    rendered_go = _draw_panel(frame, analysis, game_over=True, winner="Y")
    assert rendered_go is not None
    assert rendered_go.shape == frame.shape

    # Render with game_over = False
    rendered_live = _draw_panel(frame, analysis, game_over=False, winner=None)
    assert rendered_live is not None
    assert rendered_live.shape == frame.shape

