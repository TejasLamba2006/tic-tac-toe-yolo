"""
renderer.py — OpenCV GUI overlay for TicTacToe
Renders on top of the live camera frame (no GTK required; works on STM32MP257).

Features:
  - Live camera feed as background
  - YOLO detection boxes with labels
  - 3×3 game grid drawn on the frame
  - X / O piece labels at each cell
  - Pulsing cyan highlight for AI suggested move
  - Thick coloured line through winning triple
  - Semi-transparent HUD panel: Board state, FPS, inference time, game status
  - Win / Draw / Loss full-screen overlay with animated border
"""
from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from game_logic import EMPTY, RED, YELLOW, Board


# ---------------------------------------------------------------------------
# Colour palette  (BGR)
# ---------------------------------------------------------------------------
CLR_GRID       = (255, 230, 80)    # yellow grid lines
CLR_USER_PIECE = (80,  80,  255)   # red (BGR) for user X
CLR_AI_PIECE   = (0,   230, 255)   # cyan-yellow for AI O
CLR_SUGGEST    = (0,   255, 255)   # pulsing cyan for suggested move
CLR_WIN_LINE   = (0,   255, 0)     # bright green winning line
CLR_PANEL_BG   = (18,  18,  18)
CLR_TEXT_MAIN  = (240, 240, 240)
CLR_TEXT_DIM   = (160, 160, 160)
CLR_STATUS     = {
    "Playing":               (0,   255, 120),
    "Waiting for move":      (0,   255, 120),
    "Winning move available":(0,   215, 255),
    "Blocking move required":(0,   140, 255),
    "Forced draw":           (120, 120, 255),
    "Draw expected":         (120, 120, 255),
    "Win":                   (0,   255, 0),
    "Loss":                  (0,   0,   255),
    "Draw":                  (200, 200, 0),
    "Invalid board":         (0,   0,   200),
    "Invalid Board":         (0,   0,   200),
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _put(frame, text, pos, scale=0.55, color=CLR_TEXT_MAIN, thickness=1, bold=False):
    f = FONT_BOLD if bold else FONT
    cv2.putText(frame, text, pos, f, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, f, scale, color,    thickness,     cv2.LINE_AA)


def _text_w(text, scale=0.55, thickness=1) -> int:
    (w, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    return w


def _panel_bg(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, alpha: float = 0.6) -> None:
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi    = frame[y1:y2, x1:x2]
    dark   = np.full_like(roi, CLR_PANEL_BG)
    cv2.addWeighted(dark, alpha, roi, 1 - alpha, 0, roi)
    frame[y1:y2, x1:x2] = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Detection boxes
# ---------------------------------------------------------------------------

def draw_detections(frame: np.ndarray, detections) -> None:
    """Draw YOLO detection bounding boxes on the frame."""
    h, w = frame.shape[:2]
    label_colors = {
        "red_ball":    (60,  60,  220),
        "yellow_ball": (0,   200, 220),
        "empty":       (100, 100, 100),
    }
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        x1, y1 = int(np.clip(x1, 0, w - 1)), int(np.clip(y1, 0, h - 1))
        x2, y2 = int(np.clip(x2, 0, w - 1)), int(np.clip(y2, 0, h - 1))
        clr = label_colors.get(det.label, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), clr, 2, cv2.LINE_AA)
        label = f"{det.label.replace('_', ' ')} {det.confidence:.2f}"
        ty    = max(14, y1 - 6)
        _put(frame, label, (x1, ty), scale=0.40, color=clr)


# ---------------------------------------------------------------------------
# Grid overlay
# ---------------------------------------------------------------------------

def _grid_cell_corners(game, grid_w: int, grid_h: int, gx: int, gy: int, r: int, c: int):
    """Return (tl_x, tl_y, br_x, br_y) for a cell in the fixed fallback grid."""
    cw = grid_w // 3
    ch = grid_h // 3
    return (gx + c * cw, gy + r * ch, gx + (c + 1) * cw, gy + (r + 1) * ch)


def draw_game_overlay(
    frame:        np.ndarray,
    game,                        # TicTacToeGame instance
    detections    = None,
    user_sym:     str = RED,
    ai_sym:       str = YELLOW,
    grid_centers: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None,
    grid_radius:  int = 0,
) -> None:
    """
    Draw the game board grid + pieces + suggested move highlight.
    Uses calibrated grid if available; otherwise falls back to a fixed grid covering the centre of the frame.
    """
    h, w = frame.shape[:2]

    # Draw detection boxes underneath everything
    if detections:
        draw_detections(frame, detections)

    if grid_centers and grid_radius > 0:
        # Draw calibrated grid lines
        for r in range(3):
            cv2.line(frame, grid_centers[(r, 0)], grid_centers[(r, 2)], CLR_GRID, 2, cv2.LINE_AA)
        for c in range(3):
            cv2.line(frame, grid_centers[(0, c)], grid_centers[(2, c)], CLR_GRID, 2, cv2.LINE_AA)

        # Outer border boundaries enclosing the four corners
        p00 = grid_centers[(0, 0)]
        p02 = grid_centers[(0, 2)]
        p20 = grid_centers[(2, 0)]
        p22 = grid_centers[(2, 2)]
        cv2.line(frame, p00, p02, CLR_GRID, 2, cv2.LINE_AA)
        cv2.line(frame, p02, p22, CLR_GRID, 2, cv2.LINE_AA)
        cv2.line(frame, p22, p20, CLR_GRID, 2, cv2.LINE_AA)
        cv2.line(frame, p20, p00, CLR_GRID, 2, cv2.LINE_AA)

        # Suggested move pulsing highlight
        sug = game.suggested_move
        t   = time.time()
        if sug and not game.game_over:
            sr, sc   = sug
            cx_sug, cy_sug = grid_centers[(sr, sc)]
            pulse    = int(grid_radius * 0.85 + grid_radius * 0.15 * math.sin(t * 4))
            overlay  = frame.copy()
            cv2.circle(overlay, (cx_sug, cy_sug), pulse, CLR_SUGGEST, -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            cv2.circle(frame, (cx_sug, cy_sug), pulse, CLR_SUGGEST, 3, cv2.LINE_AA)
            _put(frame, "AI", (cx_sug - 12, cy_sug + 8), scale=0.5, color=CLR_SUGGEST, bold=True)

        # Cell labels (X / O)
        for r in range(3):
            for c in range(3):
                cell   = game.board[r][c]
                cx, cy = grid_centers[(r, c)]
                if cell == user_sym:
                    offs = int(grid_radius * 0.7)
                    cv2.line(frame, (cx - offs, cy - offs), (cx + offs, cy + offs), CLR_USER_PIECE, 6, cv2.LINE_AA)
                    cv2.line(frame, (cx + offs, cy - offs), (cx - offs, cy + offs), CLR_USER_PIECE, 6, cv2.LINE_AA)
                elif cell == ai_sym:
                    rad = int(grid_radius * 0.7)
                    cv2.circle(frame, (cx, cy), rad, CLR_AI_PIECE, 5, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (cx, cy), 4, (100, 100, 100), -1)

        # Winning line
        if game.winning_cells and len(game.winning_cells) == 3:
            pt1 = grid_centers[game.winning_cells[0]]
            pt2 = grid_centers[game.winning_cells[2]]
            cv2.line(frame, pt1, pt2, CLR_WIN_LINE, 5, cv2.LINE_AA)

    else:
        # Fallback grid: centre of frame
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.05)
        gx, gy   = margin_x, margin_y
        gw       = w - 2 * margin_x
        gh       = h - 2 * margin_y
        cw       = gw // 3
        ch       = gh // 3

        # Grid lines
        for i in range(1, 3):
            cv2.line(frame, (gx + i * cw, gy), (gx + i * cw, gy + gh), CLR_GRID, 2, cv2.LINE_AA)
            cv2.line(frame, (gx, gy + i * ch), (gx + gw, gy + i * ch), CLR_GRID, 2, cv2.LINE_AA)

        # Outer border
        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), CLR_GRID, 2, cv2.LINE_AA)

        # Suggested move pulsing highlight
        sug = game.suggested_move
        t   = time.time()
        if sug and not game.game_over:
            sr, sc   = sug
            cx_sug   = gx + sc * cw + cw // 2
            cy_sug   = gy + sr * ch + ch // 2
            pulse    = int(cw * 0.35 + cw * 0.08 * math.sin(t * 4))
            overlay  = frame.copy()
            cv2.circle(overlay, (cx_sug, cy_sug), pulse, CLR_SUGGEST, -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            cv2.circle(frame, (cx_sug, cy_sug), pulse, CLR_SUGGEST, 3, cv2.LINE_AA)
            _put(frame, "AI", (cx_sug - 12, cy_sug + 8), scale=0.5, color=CLR_SUGGEST, bold=True)

        # Cell labels (X / O)
        for r in range(3):
            for c in range(3):
                cell   = game.board[r][c]
                cx     = gx + c * cw + cw // 2
                cy     = gy + r * ch + ch // 2
                if cell == user_sym:
                    offs = int(cw * 0.28)
                    cv2.line(frame, (cx - offs, cy - offs), (cx + offs, cy + offs), CLR_USER_PIECE, 6, cv2.LINE_AA)
                    cv2.line(frame, (cx + offs, cy - offs), (cx - offs, cy + offs), CLR_USER_PIECE, 6, cv2.LINE_AA)
                elif cell == ai_sym:
                    rad = int(cw * 0.28)
                    cv2.circle(frame, (cx, cy), rad, CLR_AI_PIECE, 5, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (cx, cy), 4, (100, 100, 100), -1)

        # Winning line
        if game.winning_cells and len(game.winning_cells) == 3:
            def cell_centre(r, c):
                return (gx + c * cw + cw // 2, gy + r * ch + ch // 2)
            pt1 = cell_centre(*game.winning_cells[0])
            pt2 = cell_centre(*game.winning_cells[2])
            cv2.line(frame, pt1, pt2, CLR_WIN_LINE, 5, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# HUD panel  (top-left semi-transparent panel)
# ---------------------------------------------------------------------------

def draw_hud(
    frame:          np.ndarray,
    game,
    fps:            float,
    inference_ms:   float,
    user_sym:       str   = RED,
    ai_sym:         str   = YELLOW,
    board_rotation: int   = 0,
) -> None:
    """
    Draw a semi-transparent info panel on the top-left of the frame.
    """
    h, w     = frame.shape[:2]
    panel_w  = min(340, w // 2)
    panel_h  = 280
    _panel_bg(frame, 10, 10, 10 + panel_w, 10 + panel_h)

    y  = 34
    lh = 26   # line height

    # Title
    _put(frame, "TIC-TAC-TOE", (20, y), scale=0.7, color=(255, 220, 60), bold=True)
    y += lh + 4

    # Board grid text
    _put(frame, "Board:", (20, y), scale=0.50, color=CLR_TEXT_DIM)
    y += 20
    for row in game.board:
        def s(cell):
            if cell == user_sym:  return "X"
            if cell == ai_sym:    return "O"
            return "."
        txt = f"  {s(row[0])} | {s(row[1])} | {s(row[2])}"
        _put(frame, txt, (20, y), scale=0.52, color=CLR_TEXT_MAIN)
        y += 20
    y += 4

    # Timing
    _put(frame, f"FPS: {fps:5.1f}", (20, y), scale=0.50, color=CLR_TEXT_DIM)
    y += lh
    _put(frame, f"Infer: {inference_ms:5.1f} ms", (20, y), scale=0.50, color=CLR_TEXT_DIM)
    y += lh

    # Rotation
    _put(frame, f"Board rotation: {board_rotation}°", (20, y), scale=0.48, color=CLR_TEXT_DIM)
    y += lh

    # Turn
    turn_txt = "Turn: User (X)" if game.current_player == user_sym else "Turn: Robot (O)"
    _put(frame, turn_txt, (20, y), scale=0.55, color=CLR_TEXT_MAIN, bold=True)
    y += lh

    # Status
    raw_status  = game.game_status_msg or "Playing"
    # Normalise long messages to short display form
    if "Invalid" in raw_status:
        disp = "Invalid Board"
    elif "Forced" in raw_status or "draw" in raw_status.lower():
        disp = "Forced Draw"
    elif "Winning" in raw_status:
        disp = "Winning Move!"
    elif "Blocking" in raw_status:
        disp = "Block Required"
    else:
        disp = raw_status

    clr_st = CLR_STATUS.get(raw_status, CLR_STATUS.get(disp, CLR_TEXT_MAIN))
    _put(frame, f"Status: {disp}", (20, y), scale=0.55, color=clr_st, bold=True)
    y += lh

    # Suggested move
    if game.suggested_move and not game.game_over:
        r, c = game.suggested_move
        _put(frame, f"Best move: row={r} col={c}", (20, y), scale=0.50, color=CLR_SUGGEST)
    elif game.game_over:
        if game.winner:
            who = "Robot wins!" if game.winner == ai_sym else "User wins!"
            _put(frame, who, (20, y), scale=0.55, color=CLR_WIN_LINE, bold=True)
        else:
            _put(frame, "Draw!", (20, y), scale=0.55, color=(200, 200, 80), bold=True)

    # ---------------------------------------------------------------
    # Bottom-right FPS badge
    # ---------------------------------------------------------------
    fps_txt = f"FPS {fps:.0f}"
    fps_w   = _text_w(fps_txt, scale=0.6, thickness=2)
    _put(frame, fps_txt, (w - fps_w - 14, h - 14), scale=0.60, color=(60, 220, 120), thickness=2)

    # ---------------------------------------------------------------
    # Bottom-left: board rotation hint
    # ---------------------------------------------------------------
    _put(frame, "Press B: rotate board 90°", (8, h - 14), scale=0.45, color=CLR_TEXT_DIM)


# ---------------------------------------------------------------------------
# Win / draw full-screen overlay
# ---------------------------------------------------------------------------

def draw_win_message(frame: np.ndarray, game, duration: float = 6.0) -> None:
    """Overlay a semi-transparent win/draw/loss message (fades after *duration* s)."""
    if not game.game_just_ended:
        return

    elapsed = time.time() - game.win_message_time
    if elapsed > duration:
        game.game_just_ended = False
        return

    h, w = frame.shape[:2]

    # Dark vignette
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 10), -1)
    alpha = 0.60 - 0.08 * elapsed
    cv2.addWeighted(overlay, max(0.0, alpha), frame, 1 - max(0.0, alpha), 0, frame)

    # Main text
    if game.winner == game.ai_sym:
        msg, clr = "ROBOT WINS!",  (0, 215, 255)
    elif game.winner == game.user_sym:
        msg, clr = "YOU WIN!",     (60, 255, 60)
    elif "Forced" in game.game_status_msg:
        msg, clr = "FORCED DRAW",  (180, 180, 80)
    else:
        msg, clr = "DRAW!",        (200, 200, 60)

    scale = 2.0
    tw    = _text_w(msg, scale=scale, thickness=4)
    tx    = (w - tw) // 2
    ty    = h // 2 - 10
    cv2.putText(frame, msg, (tx + 3, ty + 3), FONT_BOLD, scale, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(frame, msg, (tx,     ty),     FONT_BOLD, scale, clr,       4, cv2.LINE_AA)

    # Sub text
    sub = "Press 'r' to restart"
    sw  = _text_w(sub, scale=0.7)
    cv2.putText(frame, sub, ((w - sw) // 2, ty + 50), FONT, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

    # Animated border
    border_t = int(5 + 3 * abs(math.sin(elapsed * 3)))
    cv2.rectangle(frame, (18, 18), (w - 18, h - 18), clr, border_t, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Zoom / pan info strip (bottom-right)
# ---------------------------------------------------------------------------

def draw_zoom_pan_info(frame: np.ndarray, camera_controller) -> None:
    h, w = frame.shape[:2]
    info = [
        f"Zoom: {camera_controller.zoom_factor:.1f}x",
        f"Pan:  ({int(camera_controller.pan_x)}, {int(camera_controller.pan_y)})",
        f"Flip: {'ON' if camera_controller.flip_upside_down else 'OFF'}",
    ]
    x_base = w - 200
    y_base = h - 80
    for i, txt in enumerate(info):
        _put(frame, txt, (x_base, y_base + i * 22), scale=0.50, color=CLR_TEXT_DIM)


# ---------------------------------------------------------------------------
# Calibration HUD & Overlay
# ---------------------------------------------------------------------------

def draw_calibration_overlay(
    frame:           np.ndarray,
    mode:            int,
    detected_count:  int,
    grid_centers:    Optional[Dict[Tuple[int, int], Tuple[int, int]]],
    grid_radius:     int,
    stable_count:    int,
    required_stable: int,
) -> None:
    """Draw a dedicated HUD and preview lines during grid calibration."""
    h, w = frame.shape[:2]

    # Draw info panel at top left
    _panel_bg(frame, 10, 10, 360, 160)

    _put(frame, "GRID CALIBRATION", (20, 35), scale=0.65, color=(0, 255, 255), bold=True)
    _put(frame, f"Detected Balls: {detected_count} / 5", (20, 68), scale=0.52, color=CLR_TEXT_MAIN)

    if grid_centers and grid_radius > 0:
        # Draw grid lines between cell centers
        for r in range(3):
            cv2.line(frame, grid_centers[(r, 0)], grid_centers[(r, 2)], (0, 255, 0), 2, cv2.LINE_AA)
        for c in range(3):
            cv2.line(frame, grid_centers[(0, c)], grid_centers[(2, c)], (0, 255, 0), 2, cv2.LINE_AA)

        # Enclosing border around four corners
        p00 = grid_centers[(0, 0)]
        p02 = grid_centers[(0, 2)]
        p20 = grid_centers[(2, 0)]
        p22 = grid_centers[(2, 2)]
        cv2.line(frame, p00, p02, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(frame, p02, p22, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(frame, p22, p20, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(frame, p20, p00, (0, 255, 0), 2, cv2.LINE_AA)

        # Cell markers & radii
        for (r, c), (gx, gy) in grid_centers.items():
            cv2.circle(frame, (gx, gy), 5, (0, 255, 0), -1)
            cv2.circle(frame, (gx, gy), grid_radius, (0, 255, 0), 1, cv2.LINE_AA)

        _put(frame, "Grid found! Stabilising...", (20, 100), scale=0.50, color=(0, 255, 0))
        _put(frame, f"Stability: {stable_count}/{required_stable} frames", (20, 130), scale=0.50, color=(0, 255, 0))
    else:
        _put(frame, "Place balls at 4 corners + center", (20, 100), scale=0.48, color=(80, 80, 255))
        _put(frame, "Waiting for exactly 5 balls...", (20, 130), scale=0.48, color=(80, 80, 255))

