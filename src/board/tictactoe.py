#!/usr/bin/env python3
"""
tictactoe.py — STM32MP257 TicTacToe main loop (YOLO + OpenCV GUI overlay)

Pipeline:
  camera frame → YOLO inference → board state estimation → board validation
               → Minimax suggestion → OpenCV overlay → display
               → robot arm command (when AI turn confirmed)

Set TICTACTOE_DEBUG=1 for verbose pipeline logging.

Usage:
  python3 tictactoe.py --weights best.pt [--user-color R] [--ai-color Y]
                       [--camera auto] [--npu] [--width 640] [--height 480]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Local modules
import comms
import camera_controller
import renderer
from vision import (
    EMPTY, RED, YELLOW,
    VisionSystem,
    board_counts,
    board_to_game_symbols,
    game_symbols_to_board,
)
from game_logic import TicTacToeGame, validate_board


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class TicTacToeApp:
    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg = cfg

        # Color assignment (CLI configurable)
        self.user_sym = cfg.user_color   # 'R' or 'Y'
        self.ai_sym   = cfg.ai_color     # 'Y' or 'R'
        assert self.user_sym != self.ai_sym, "--user-color and --ai-color must differ"

        # Sub-systems
        self.vision = VisionSystem(
            weights_path          = cfg.weights,
            width                 = cfg.width,
            height                = cfg.height,
            fps                   = cfg.fps,
            confidence_threshold  = cfg.confidence,
            iou_threshold         = cfg.iou,
            image_size            = cfg.image_size,
            smoothing_window      = cfg.smoothing,
            use_npu               = cfg.npu,
        )
        self.cam_ctrl  = camera_controller.CameraController(cfg.width, cfg.height)
        self.game      = TicTacToeGame(user_sym=self.user_sym, ai_sym=self.ai_sym)

        # UI IPC
        self._ui_started = False
        self.should_exit = False

        # Stability gate — board must be stable N frames before we act
        self.required_stable_frames   = cfg.stable_frames
        self._stable_count            = 0
        self._last_stable_board: List[List[str]] = [[EMPTY] * 3 for _ in range(3)]

        # User-move confirmation state
        self._pending_user_move: Optional[Tuple[int, int]] = None
        self._pending_user_stable = 0
        self._user_move_required_stable = cfg.stable_frames

        # AI move confirmation state (wait for YOLO to see the ball placed)
        self._waiting_ai_confirm        = False
        self._ai_confirm_stable         = 0
        self._ai_confirm_required       = 20
        self._ai_target_cell: Optional[Tuple[int, int]] = None

        # Computer move delay after user moves
        self._waiting_computer_turn     = False
        self._computer_move_time        = 0.0
        self._computer_move_delay       = cfg.ai_delay

        # Game flow flags
        self.board_needs_clearing = False
        self._board_clear_count   = 0
        self.mode = 0  # 0=preview/game (combined), 1=paused

        # FPS tracking
        self._fps_start    = time.time()
        self._fps_count    = 0
        self._current_fps  = 0.0

        # Board flip rotation (toggled by 'b' key or GUI button)
        self._board_rotation = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        print("=" * 52)
        print("   STM32MP257 TIC-TAC-TOE  (YOLO + OpenCV GUI)")
        print("=" * 52)
        print(f"   User  = {self.user_sym}   AI = {self.ai_sym}")
        print(f"   Weights: {self.cfg.weights}")
        print(f"   NPU: {'enabled' if self.cfg.npu else 'disabled'}")
        print("=" * 52)

        comms.init_move_process()

        if not self.vision.setup_yolo():
            print("ERROR: Failed to load YOLO model!")
            return False

        if not self.vision.setup_camera():
            print("ERROR: Failed to open camera!")
            return False

        return True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        window = "TicTacToe — STM32MP257"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        print("\n[KEYS]  q/ESC=quit  r=reset  b=rotate board  p=pause  m=manual")
        print(f"[INFO]  TICTACTOE_DEBUG={os.environ.get('TICTACTOE_DEBUG', '0')}\n")

        comms.send_ui_update("start")
        comms.send_ui_update("turn X")

        manual_mode = False

        while not self.should_exit:
            ok, frame = self.vision.get_frame()
            if not ok or frame is None:
                cv2.waitKey(10)
                continue

            # Camera transform (zoom / pan / flip)
            frame = self.cam_ctrl.apply_transform(frame)
            if frame is None:
                continue

            # ---- YOLO inference ----------------------------------------
            detections, raw_board = self.vision.run_inference(frame)
            # Apply user's board-rotation preference
            raw_board = self._rotate_board(raw_board)

            # ---- Board stability gate ----------------------------------
            board_changed = (raw_board != self._last_stable_board)
            if board_changed:
                self._stable_count      = 0
                self._last_stable_board = [row[:] for row in raw_board]
            else:
                self._stable_count += 1

            board_stable = (self._stable_count >= self.required_stable_frames)

            # ---- Validate board ----------------------------------------
            valid, inv_reason = validate_board(raw_board, self.user_sym, self.ai_sym)
            if not valid:
                self.game.game_status_msg = f"Invalid Board"
                self.game.suggested_move  = None
                if os.environ.get("TICTACTOE_DEBUG"):
                    print(f"[VALIDATE] INVALID: {inv_reason}")
            else:
                if board_stable and not self.game.game_over:
                    self._process_stable_board(raw_board, manual_mode)

            # ---- Update Minimax suggestion ----------------------------
            self.game.update_suggestion()

            # ---- AI move timing ---------------------------------------
            self._tick_computer_move()

            # ---- AI move confirmation ---------------------------------
            self._tick_ai_confirm(raw_board)

            # ---- Game end check ---------------------------------------
            self.game.check_game_end()
            self.game.log_status()

            # ---- Render -----------------------------------------------
            display = frame.copy()
            renderer.draw_game_overlay(
                display, self.game,
                detections=detections,
                user_sym=self.user_sym,
                ai_sym=self.ai_sym,
            )
            renderer.draw_hud(
                display, self.game,
                fps=self._current_fps,
                inference_ms=self.vision.last_inference_ms,
                user_sym=self.user_sym,
                ai_sym=self.ai_sym,
                board_rotation=self._board_rotation,
            )
            renderer.draw_win_message(display, self.game)
            renderer.draw_zoom_pan_info(display, self.cam_ctrl)

            # Stability indicator
            if not board_stable and not self.game.game_over:
                cv2.putText(
                    display,
                    f"Stabilising {self._stable_count}/{self.required_stable_frames}...",
                    (10, display.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 0), 1, cv2.LINE_AA,
                )

            cv2.imshow(window, display)

            # ---- FPS tracking ----------------------------------------
            self._fps_count += 1
            if self._fps_count >= 30:
                elapsed = time.time() - self._fps_start
                self._current_fps = 30 / elapsed if elapsed > 0 else 0.0
                self._fps_start   = time.time()
                self._fps_count   = 0

            # ---- Start UI subprocess once ----------------------------
            if not self._ui_started:
                self._start_ui()

            # ---- Key handling ----------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("r"):
                self._reset_game()
            elif key == ord("b"):
                # Rotate board 90° CW
                self._board_rotation = (self._board_rotation + 90) % 360
                print(f"[APP] Board rotation: {self._board_rotation}°")
            elif key == ord("p"):
                self.mode = 1 - self.mode
                print(f"[APP] {'Paused' if self.mode == 1 else 'Resumed'}")
            elif key == ord("m"):
                manual_mode = not manual_mode
                print(f"[APP] Manual mode: {manual_mode}")
            elif key == ord("=") or key == ord("+"):
                if self.cam_ctrl.zoom_in(): self._sync_trackbars()
            elif key == ord("-"):
                if self.cam_ctrl.zoom_out(): self._sync_trackbars()
            elif key == 82:  self.cam_ctrl.pan_up();    self._sync_trackbars()
            elif key == 84:  self.cam_ctrl.pan_down();  self._sync_trackbars()
            elif key == 81:  self.cam_ctrl.pan_left();  self._sync_trackbars()
            elif key == 83:  self.cam_ctrl.pan_right(); self._sync_trackbars()
            elif key == ord("f"):
                self.cam_ctrl.toggle_flip()

        # Cleanup
        self.vision.cleanup()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Board rotation helper
    # ------------------------------------------------------------------

    def _rotate_board(self, board: List[List[str]]) -> List[List[str]]:
        rotations = self._board_rotation // 90
        result = board
        for _ in range(rotations):
            result = [list(row) for row in zip(*result[::-1])]
        return result

    # ------------------------------------------------------------------
    # Game flow
    # ------------------------------------------------------------------

    def _process_stable_board(self, detected_board: List[List[str]], manual_mode: bool) -> None:
        """
        Compare detected_board against game internal board and decide:
          - If user placed a new ball → confirm and hand off to AI
          - If AI move already visible → confirm robot arm placement
        """
        if self.game.game_over:
            return
        if self.mode == 1:
            return  # paused

        event = self.game.sync_from_detection(detected_board)

        if os.environ.get("TICTACTOE_DEBUG"):
            print(f"[FLOW] event={event} player={self.game.current_player} "
                  f"waiting_AI={self._waiting_computer_turn}")

        if event == "user_move" and self.game.current_player == self.user_sym:
            if not manual_mode or self._stable_count >= self._user_move_required_stable:
                self._confirm_user_move(detected_board)

        elif event == "ai_confirmed" and self._waiting_ai_confirm:
            self._confirm_ai_placement(detected_board)

        elif event == "invalid":
            self.game.game_status_msg = "Invalid Board"
            self.game.suggested_move  = None

    def _confirm_user_move(self, detected_board: List[List[str]]) -> None:
        """Apply the newly detected user move to the internal board."""
        # Find the new user cell
        for r in range(3):
            for c in range(3):
                if (self.game.board[r][c] == EMPTY
                        and detected_board[r][c] == self.user_sym):
                    print(f"[APP] ✅ User move confirmed: ({r},{c})")
                    self.game.board[r][c]  = self.user_sym
                    self.game.current_player = self.ai_sym
                    self._stable_count     = 0
                    comms.send_ui_update(f"move {r} {c} X")
                    comms.send_ui_update("turn O")

                    if not self.game.check_game_end():
                        # Schedule AI response
                        self._waiting_computer_turn = True
                        self._computer_move_time    = time.time() + self._computer_move_delay
                    return

    def _tick_computer_move(self) -> None:
        """Fire AI move when delay has elapsed."""
        if (not self._waiting_computer_turn
                or self.game.game_over
                or time.time() < self._computer_move_time):
            return

        self._waiting_computer_turn = False
        # Refresh suggestion
        self.game.update_suggestion()

        if self.game.suggested_move is None:
            print("[APP] No AI move available.")
            return

        row, col = self.game.suggested_move
        print(f"[APP] 🤖 AI move: ({row},{col})  [{self.game.suggested_status}]")

        # Send robot arm command
        map_grid = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        comms.send(str(map_grid[row][col]))

        # Update internal board immediately
        self.game.board[row][col] = self.ai_sym
        self.game.current_player  = self.user_sym
        self.game.suggested_move  = None

        comms.send_ui_update(f"move {row} {col} O")
        comms.send_ui_update("turn X")

        # Start visual confirmation (wait for YOLO to see the ball)
        self._waiting_ai_confirm = True
        self._ai_target_cell     = (row, col)
        self._ai_confirm_stable  = 0

        self.game.check_game_end()

    def _tick_ai_confirm(self, detected: List[List[str]]) -> None:
        """Count how many frames YOLO sees the AI's ball in the expected cell."""
        if not self._waiting_ai_confirm or self._ai_target_cell is None:
            return
        r, c = self._ai_target_cell
        if detected[r][c] == self.ai_sym:
            self._ai_confirm_stable += 1
        else:
            self._ai_confirm_stable = max(0, self._ai_confirm_stable - 1)

        if self._ai_confirm_stable >= self._ai_confirm_required:
            print(f"[APP] ✅ AI placement confirmed at ({r},{c})")
            self._waiting_ai_confirm = False
            self._ai_target_cell     = None
            self._ai_confirm_stable  = 0

    def _confirm_ai_placement(self, detected: List[List[str]]) -> None:
        pass  # handled by _tick_ai_confirm

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_game(self) -> None:
        print("[APP] 🔄 Game reset")
        self.game.reset()
        self.vision.reset_estimator()
        self._stable_count             = 0
        self._last_stable_board        = [[EMPTY] * 3 for _ in range(3)]
        self._waiting_computer_turn    = False
        self._waiting_ai_confirm       = False
        self._ai_target_cell           = None
        self._ai_confirm_stable        = 0
        self._pending_user_move        = None
        self._pending_user_stable      = 0

    # ------------------------------------------------------------------
    # UI subprocess
    # ------------------------------------------------------------------

    def _start_ui(self) -> None:
        self._ui_started = True
        try:
            comms.app = subprocess.Popen(
                ["python3", "ui.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
            )
            threading.Thread(
                target=comms.listen_feedback,
                args=(self,),
                daemon=True,
            ).start()
            comms.flush_ui_queue()
        except Exception as e:
            print(f"[APP] UI subprocess failed (non-fatal): {e}")

    def _sync_trackbars(self) -> None:
        pass  # trackbars removed; camera controlled via keyboard


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TicTacToe — YOLO + STM32MP257")
    p.add_argument("--weights",      default="best.pt",
                   help="YOLO weights: .pt / .onnx / .tflite / .nb")
    p.add_argument("--camera",       default="auto",
                   help="Camera source: auto, index (0,1…), /dev/videoX, csi, usb")
    p.add_argument("--user-color",   default="R",  choices=("R", "Y"),
                   help="Color of user's balls (R=red, Y=yellow). Default: R")
    p.add_argument("--ai-color",     default="Y",  choices=("R", "Y"),
                   help="Color of robot's balls.  Default: Y")
    p.add_argument("--width",        type=int,   default=640)
    p.add_argument("--height",       type=int,   default=480)
    p.add_argument("--fps",          type=float, default=30.0)
    p.add_argument("--confidence",   type=float, default=0.50,
                   help="YOLO confidence threshold")
    p.add_argument("--iou",          type=float, default=0.45,
                   help="YOLO IoU NMS threshold")
    p.add_argument("--image-size",   type=int,   default=320,
                   help="YOLO input resolution")
    p.add_argument("--smoothing",    type=int,   default=5,
                   help="Board-state smoothing window (frames)")
    p.add_argument("--stable-frames",type=int,   default=10,
                   help="Frames board must be stable before acting")
    p.add_argument("--ai-delay",     type=float, default=1.5,
                   help="Seconds to wait before AI sends its move command")
    p.add_argument("--npu",          action="store_true",
                   help="Enable STM32 NPU hardware acceleration")
    return p


def main() -> None:
    parser = build_parser()
    cfg    = parser.parse_args()

    # Validate color assignment
    if cfg.user_color == cfg.ai_color:
        parser.error("--user-color and --ai-color must be different")

    app = TicTacToeApp(cfg)
    if app.initialize():
        app.run()
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
