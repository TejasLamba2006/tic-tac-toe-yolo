"""
vision.py — YOLO-based vision system for TicTacToe
Replaces the old HSV color-range pipeline with proper YOLO detection.
Handles camera init (DCMIPP / V4L2 / webcam), frame capture threading,
board state estimation with temporal smoothing, and pipeline logging.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from collections import Counter, deque
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.vision.board_state import Detection
from src.ai.yolo_inference import YoloInference


# ---------------------------------------------------------------------------
# Board state constants
# ---------------------------------------------------------------------------
EMPTY   = "E"   # empty cell
RED     = "R"   # red ball  (human)
YELLOW  = "Y"   # yellow ball (robot)


# ---------------------------------------------------------------------------
# Board state estimator  (temporal smoothing, adapted from reference project)
# ---------------------------------------------------------------------------

class BoardStateEstimator:
    """
    Convert raw YOLO detections into a stable 3×3 board.

    The model predicts one box per object.  Detections are assigned to the
    nearest grid cell and a per-cell vote history smooths out noisy frames.
    """

    LABEL_TO_SYMBOL = {
        "empty":       EMPTY,
        "red_ball":    RED,
        "yellow_ball": YELLOW,
    }

    def __init__(
        self,
        minimum_confidence: float = 0.45,
        smoothing_window:   int   = 5,
        stability_ratio:    float = 0.6,
    ) -> None:
        self.minimum_confidence = minimum_confidence
        self.stability_ratio    = stability_ratio

        history_size = max(1, smoothing_window)
        self._cell_history: List[List[deque]] = [
            [deque(maxlen=history_size) for _ in range(3)]
            for _ in range(3)
        ]
        self._stable_board: List[List[str]] = [[EMPTY] * 3 for _ in range(3)]
        self._initialized = False
        self.grid_centers: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.grid_radius: int = 0

    def reset(self) -> None:
        for row in self._cell_history:
            for q in row:
                q.clear()
        self._stable_board = [[EMPTY] * 3 for _ in range(3)]
        self._initialized = False

    def estimate(
        self,
        detections: List[Detection],
        frame_w: int,
        frame_h: int,
    ) -> List[List[str]]:
        """
        Return the (smoothed) 3×3 board given the current frame's detections.

        frame_w / frame_h are used to compute cell boundaries when the model
        operates on the full camera frame (no perspective warp step).
        """
        instant = self._build_instant(detections, frame_w, frame_h)
        return self._stabilize(instant)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_instant(
        self,
        detections: List[Detection],
        fw: int,
        fh: int,
    ) -> List[List[str]]:
        board      = [[EMPTY] * 3 for _ in range(3)]
        priorities = [[(-1, 0.0)] * 3 for _ in range(3)]

        cell_w = fw / 3.0
        cell_h = fh / 3.0

        for det in detections:
            symbol = self.LABEL_TO_SYMBOL.get(det.label)
            if symbol is None or det.confidence < self.minimum_confidence:
                continue

            cx, cy = det.center

            if self.grid_centers and self.grid_radius > 0:
                best_cell = None
                best_dist = float('inf')
                for (r, c), (gx, gy) in self.grid_centers.items():
                    dist = np.sqrt((cx - gx)**2 + (cy - gy)**2)
                    if dist < self.grid_radius and dist < best_dist:
                        best_dist = dist
                        best_cell = (r, c)
                if best_cell is not None:
                    row, col = best_cell
                else:
                    # Outside board grid boundaries, ignore the detection
                    continue
            else:
                col = max(0, min(2, int(cx / cell_w)))
                row = max(0, min(2, int(cy / cell_h)))

            priority = (1 if symbol != EMPTY else 0, det.confidence)
            if priority >= priorities[row][col]:
                board[row][col]      = symbol
                priorities[row][col] = priority

        return board

    def calibrate_grid(self, centroids: List[Tuple[float, float]]) -> bool:
        """
        Attempt to form a 3x3 grid from 5 points (Corners + Center).
        Expected points: (0,0), (0,2), (1,1), (2,0), (2,2)
        """
        if len(centroids) != 5:
            return False

        try:
            # Sort by Y (top to bottom)
            centroids_sorted = sorted(centroids, key=lambda p: p[1])
            
            # Top row (2 points)
            top = sorted(centroids_sorted[:2], key=lambda p: p[0])
            # Middle (1 point)
            mid = centroids_sorted[2]
            # Bottom row (2 points)
            bottom = sorted(centroids_sorted[3:], key=lambda p: p[0])

            p00, p02 = top[0], top[1]
            p11 = mid
            p20, p22 = bottom[0], bottom[1]

            # Interpolate missing points
            p01 = ((p00[0] + p02[0]) / 2.0, (p00[1] + p02[1]) / 2.0)
            p10 = ((p00[0] + p20[0]) / 2.0, (p00[1] + p20[1]) / 2.0)
            p12 = ((p02[0] + p22[0]) / 2.0, (p02[1] + p22[1]) / 2.0)
            p21 = ((p20[0] + p22[0]) / 2.0, (p20[1] + p22[1]) / 2.0)

            def to_int_pt(pt):
                return (int(pt[0]), int(pt[1]))

            self.grid_centers = {
                (0, 0): to_int_pt(p00), (0, 1): to_int_pt(p01), (0, 2): to_int_pt(p02),
                (1, 0): to_int_pt(p10), (1, 1): to_int_pt(p11), (1, 2): to_int_pt(p12),
                (2, 0): to_int_pt(p20), (2, 1): to_int_pt(p21), (2, 2): to_int_pt(p22)
            }

            dist_00_01 = np.linalg.norm(np.array(p00) - np.array(p01))
            self.grid_radius = int(dist_00_01 * 0.4)
            return True
        except Exception as e:
            print(f"[VISION] Grid calibration exception: {e}")
            return False

    def _stabilize(self, instant: List[List[str]]) -> List[List[str]]:
        if not self._initialized:
            self._initialized  = True
            self._stable_board = [row[:] for row in instant]

        history_size   = self._cell_history[0][0].maxlen or 1
        required_votes = max(2, ceil(history_size * self.stability_ratio))
        next_board     = [row[:] for row in self._stable_board]

        for r in range(3):
            for c in range(3):
                history = self._cell_history[r][c]
                history.append(instant[r][c])
                counts = Counter(history)

                max_votes = max(counts.values())
                # pick the *most recent* label that has the max vote count
                candidate = history[-1]
                for item in reversed(history):
                    if counts[item] == max_votes:
                        candidate = item
                        break

                current        = next_board[r][c]
                current_votes  = counts.get(current, 0)
                new_votes      = counts.get(candidate, 0)
                if candidate != current and (
                    (new_votes > current_votes and new_votes >= required_votes)
                    or (current_votes <= 1 and new_votes == current_votes)
                ):
                    next_board[r][c] = candidate

        self._stable_board = next_board
        return [row[:] for row in self._stable_board]


# ---------------------------------------------------------------------------
# VisionSystem
# ---------------------------------------------------------------------------

class VisionSystem:
    """
    Camera setup + YOLO inference + board state estimation.

    Detection pipeline:
      camera frame  →  YoloInference.predict()  →  BoardStateEstimator.estimate()
                   →  List[Detection]  +  board[3][3]
    """

    def __init__(
        self,
        weights_path: str,
        width:  int   = 640,
        height: int   = 480,
        fps:    float = 30.0,
        confidence_threshold: float = 0.50,
        iou_threshold:        float = 0.45,
        image_size:           int   = 320,
        smoothing_window:     int   = 5,
        use_npu:              bool  = False,
        device:               Optional[str] = None,
    ) -> None:
        self.weights_path = weights_path
        self.width   = width
        self.height  = height
        self.fps     = fps

        # YOLO
        self.yolo: Optional[YoloInference] = None
        self.confidence_threshold = confidence_threshold
        self.iou_threshold        = iou_threshold
        self.image_size           = image_size
        self.use_npu              = use_npu
        self.device               = device

        # Board estimator
        self.estimator = BoardStateEstimator(
            minimum_confidence=confidence_threshold,
            smoothing_window=smoothing_window,
        )

        # Camera
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_device: Optional[str]    = None
        self.camera_type   = "unknown"
        self.dcmipp_media  = "platform:48030000.dcmipp"

        # Threaded frame grab
        self.use_threading   = True
        self.stop_thread     = False
        self.frame_lock      = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_thread:  Optional[threading.Thread] = None
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10

        # Timing
        self.last_inference_ms: float = 0.0

        # Board rotation state (0 / 90 / 180 / 270 degrees)
        self.board_rotation: int = 0   # incremented by 90 on each GUI btn press

    # ------------------------------------------------------------------
    # YOLO init
    # ------------------------------------------------------------------

    def setup_yolo(self) -> bool:
        try:
            print(f"[YOLO] Loading model: {self.weights_path}")
            self.yolo = YoloInference(
                weights_path=self.weights_path,
                confidence_threshold=self.confidence_threshold,
                iou_threshold=self.iou_threshold,
                image_size=self.image_size,
                device=self.device,
                use_npu=self.use_npu,
            )
            print("[YOLO] Model loaded successfully.")
            return True
        except Exception as e:
            print(f"[YOLO] Failed to load model: {e}")
            return False

    # ------------------------------------------------------------------
    # Camera discovery helpers
    # ------------------------------------------------------------------

    def _check_dcmipp_camera(self) -> bool:
        try:
            result = subprocess.run(
                ["media-ctl", "-d", self.dcmipp_media, "-p"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return False
            print("✓ DCMIPP media controller found")
            video_dirs = subprocess.run(
                ["find", "/sys/class/video4linux", "-name", "video*", "-type", "l"],
                capture_output=True, text=True,
            ).stdout.strip().split("\n")
            for vd in video_dirs:
                if not vd:
                    continue
                name_file = os.path.join(vd, "name")
                if os.path.exists(name_file):
                    with open(name_file) as f:
                        dev_name = f.read().strip()
                    if dev_name == "dcmipp_main_capture":
                        self.camera_device = f"/dev/{os.path.basename(vd)}"
                        print(f"✓ DCMIPP capture device: {self.camera_device}")
                        return True
        except Exception as e:
            print(f"DCMIPP check error: {e}")
        return False

    def _check_webcam(self) -> bool:
        try:
            video_dirs = subprocess.run(
                ["find", "/sys/class/video4linux", "-name", "video*", "-type", "l"],
                capture_output=True, text=True,
            ).stdout.strip().split("\n")
            for vd in video_dirs:
                if not vd:
                    continue
                name_file = os.path.join(vd, "name")
                if os.path.exists(name_file):
                    with open(name_file) as f:
                        dev_name = f.read().strip()
                    if "dcmi" not in dev_name.lower() and "stm" not in dev_name.lower():
                        self.camera_device = f"/dev/{os.path.basename(vd)}"
                        print(f"✓ Webcam: {self.camera_device} ({dev_name})")
                        return True
        except Exception as e:
            print(f"Webcam check error: {e}")
        return False

    def _try_integer_cameras(self) -> bool:
        for idx in range(10):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ok, _ = cap.read()
                cap.release()
                if ok:
                    self.camera_device = str(idx)
                    self.camera_type   = "index"
                    print(f"✓ Camera index {idx} found")
                    return True
            cap.release()
        return False

    # ------------------------------------------------------------------
    # Camera setup
    # ------------------------------------------------------------------

    def setup_camera(self) -> bool:
        print("Detecting camera hardware...")
        if self._check_dcmipp_camera():
            self.camera_type = "dcmipp"
        elif self._check_webcam():
            self.camera_type = "webcam"
        elif self._try_integer_cameras():
            pass  # camera_type already set
        else:
            print("ERROR: No camera found!")
            return False

        try:
            src = self.camera_device
            if src.isdigit() if isinstance(src, str) else False:
                self.cap = cv2.VideoCapture(int(src))
            elif isinstance(src, str) and src.startswith("/dev/"):
                self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
            else:
                self.cap = cv2.VideoCapture(src)

            if not self.cap.isOpened():
                print(f"ERROR: Cannot open camera {src}")
                return False

            # Configure — skip VIDIOC_S_FMT on V4L2 to avoid breaking DCMIPP pipeline
            if self.camera_type not in ("dcmipp", "webcam"):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS,        self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE,  1)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS,   0)

            aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"✓ Camera opened at {aw}×{ah}")
            # Sync width/height with actual camera resolution
            self.width  = aw
            self.height = ah

            if self.use_threading:
                self.stop_thread  = False
                self.frame_thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.frame_thread.start()
                print("✓ Frame capture thread started")

            return True
        except Exception as e:
            print(f"Camera setup error: {e}")
            return False

    def _capture_loop(self) -> None:
        while not self.stop_thread and self.cap and self.cap.isOpened():
            try:
                ok, frame = self.cap.read()
                if ok and frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame.copy()
                    self.consecutive_failures = 0
                else:
                    self.consecutive_failures += 1
                    if self.consecutive_failures > self.max_consecutive_failures:
                        print("Frame capture: too many consecutive failures")
                        break
                time.sleep(1.0 / (self.fps * 2))
            except Exception as e:
                print(f"Capture thread error: {e}")
                self.consecutive_failures += 1
                time.sleep(0.1)

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.use_threading:
            with self.frame_lock:
                if self.latest_frame is not None:
                    return True, self.latest_frame.copy()
                return False, None
        try:
            ok, frame = self.cap.read()
            return (ok, frame) if ok and frame is not None else (False, None)
        except Exception:
            return False, None

    # ------------------------------------------------------------------
    # YOLO inference + board estimation
    # ------------------------------------------------------------------

    def calibrate_grid(self, centroids: List[Tuple[float, float]]) -> bool:
        """Forward grid calibration to estimator."""
        return self.estimator.calibrate_grid(centroids)

    def rotate_board_90(self) -> None:
        """Rotate the logical board orientation by 90° CW (GUI button)."""
        self.board_rotation = (self.board_rotation + 90) % 360
        print(f"[VISION] Board rotation: {self.board_rotation}°")

    def _apply_board_rotation(self, board: List[List[str]]) -> List[List[str]]:
        """Rotate board matrix by self.board_rotation degrees CW."""
        rotations = self.board_rotation // 90
        result = board
        for _ in range(rotations):
            result = [list(row) for row in zip(*result[::-1])]
        return result

    def run_inference(self, frame: np.ndarray) -> Tuple[List[Detection], List[List[str]]]:
        """
        Run YOLO on *frame*, return (raw_detections, stable_board_3x3).
        Board uses EMPTY/'R'/'Y' symbols.
        Applies board_rotation transform before returning.
        """
        if self.yolo is None:
            return [], [[EMPTY] * 3 for _ in range(3)]

        t0 = time.perf_counter()
        detections = self.yolo.predict(frame)
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0

        board = self.estimator.estimate(detections, self.width, self.height)
        board = self._apply_board_rotation(board)

        # Pipeline logging
        self._log_pipeline(detections, board)

        return detections, board

    def _log_pipeline(self, detections: List[Detection], board: List[List[str]]) -> None:
        if os.environ.get("TICTACTOE_DEBUG"):
            counts = {"E": 0, "R": 0, "Y": 0}
            for det in detections:
                lbl = det.label
                if lbl == "red_ball":    counts["R"] += 1
                elif lbl == "yellow_ball": counts["Y"] += 1
                elif lbl == "empty":       counts["E"] += 1
            print(
                f"[DET] raw={len(detections)} "
                f"red={counts['R']} yellow={counts['Y']} empty={counts['E']} "
                f"infer={self.last_inference_ms:.1f}ms"
            )
            print(
                f"[BOARD]\n  {board[0][0]} | {board[0][1]} | {board[0][2]}\n"
                f"  {board[1][0]} | {board[1][1]} | {board[1][2]}\n"
                f"  {board[2][0]} | {board[2][1]} | {board[2][2]}"
            )

    def reset_estimator(self) -> None:
        """Call this when the board is cleared for a new game."""
        self.estimator.reset()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        try:
            if self.frame_thread:
                self.stop_thread = True
                self.frame_thread.join(timeout=2.0)
                print("Frame capture thread stopped")
            if self.cap and self.cap.isOpened():
                self.cap.release()
            print("Camera resources cleaned up.")
        except Exception as e:
            print(f"Cleanup error: {e}")


# ---------------------------------------------------------------------------
# Utility: count board symbols
# ---------------------------------------------------------------------------

def board_counts(board: List[List[str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {EMPTY: 0, RED: 0, YELLOW: 0}
    for row in board:
        for cell in row:
            counts[cell] = counts.get(cell, 0) + 1
    return counts


def board_to_game_symbols(board: List[List[str]], user_color: str, ai_color: str) -> List[List[str]]:
    """
    Convert E/R/Y board to ' '/'X'/'0' for TicTacToeGame.
    user_color: 'R' or 'Y'
    ai_color:   'Y' or 'R'
    """
    mapping = {EMPTY: " ", user_color: "X", ai_color: "0"}
    return [[mapping.get(cell, " ") for cell in row] for row in board]


def game_symbols_to_board(game_board: List[List[str]], user_color: str, ai_color: str) -> List[List[str]]:
    """Reverse of board_to_game_symbols."""
    mapping = {" ": EMPTY, "X": user_color, "0": ai_color}
    return [[mapping.get(cell, EMPTY) for cell in row] for row in game_board]
