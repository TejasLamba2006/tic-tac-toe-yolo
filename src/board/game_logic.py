"""
game_logic.py — TicTacToe game state, board validation, and Minimax AI
Updated to:
  - Use E/R/Y board symbols (matches YOLO pipeline)
  - Validate board before running Minimax
  - Return enriched status messages with every move decision
  - Detect: Winning move available, Blocking move required, Forced draw, Draw
  - Guard against game-over or invalid boards before Minimax
"""
from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple
from math import inf

import comms

# Symbols
EMPTY  = "E"
RED    = "R"    # default: human
YELLOW = "Y"    # default: AI / robot

Board = List[List[str]]


# ---------------------------------------------------------------------------
# Pure board functions (used by both TicTacToeGame and the move selector)
# ---------------------------------------------------------------------------

def check_winner(board: Board) -> Optional[str]:
    """Return winning symbol or None."""
    b = board
    # rows
    for row in b:
        if row[0] != EMPTY and row[0] == row[1] == row[2]:
            return row[0]
    # cols
    for c in range(3):
        if b[0][c] != EMPTY and b[0][c] == b[1][c] == b[2][c]:
            return b[0][c]
    # diagonals
    if b[0][0] != EMPTY and b[0][0] == b[1][1] == b[2][2]:
        return b[0][0]
    if b[0][2] != EMPTY and b[0][2] == b[1][1] == b[2][0]:
        return b[0][2]
    return None


def is_full(board: Board) -> bool:
    return all(cell != EMPTY for row in board for cell in row)


def is_draw(board: Board) -> bool:
    return is_full(board) and check_winner(board) is None


def empty_cells(board: Board) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(3) for c in range(3) if board[r][c] == EMPTY]


def opponent(player: str) -> str:
    return YELLOW if player == RED else RED


def board_counts(board: Board) -> dict:
    counts = {EMPTY: 0, RED: 0, YELLOW: 0}
    for row in board:
        for cell in row:
            counts[cell] = counts.get(cell, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Board validation
# ---------------------------------------------------------------------------

def validate_board(board: Board, user_sym: str = RED, ai_sym: str = YELLOW) -> Tuple[bool, str]:
    """
    Return (is_valid, reason).
    Valid if:
      - All cells are E, user_sym, or ai_sym
      - |count_user - count_ai| <= 1  (user moves first)
    """
    allowed = {EMPTY, user_sym, ai_sym}
    for row in board:
        for cell in row:
            if cell not in allowed:
                return False, f"Unknown symbol '{cell}' in board"
    counts = board_counts(board)
    u = counts.get(user_sym, 0)
    a = counts.get(ai_sym, 0)
    if abs(u - a) > 1:
        return False, f"Illegal piece counts: user={u}, ai={a} (diff={abs(u-a)} > 1)"
    return True, "OK"


# ---------------------------------------------------------------------------
# Minimax  (alpha-beta pruned)
# ---------------------------------------------------------------------------

_MOVE_ORDER = [(1, 1), (0, 0), (0, 2), (2, 0), (2, 2), (0, 1), (1, 0), (1, 2), (2, 1)]


def _ordered_moves(board: Board) -> List[Tuple[int, int]]:
    return [m for m in _MOVE_ORDER if board[m[0]][m[1]] == EMPTY]


def _minimax(
    board:    Board,
    depth:    int,
    maximizing: bool,
    ai_sym:   str,
    human_sym: str,
    alpha:    float = -inf,
    beta:     float = inf,
) -> int:
    winner = check_winner(board)
    if winner == ai_sym:
        return 10 - depth
    if winner == human_sym:
        return depth - 10
    if is_full(board):
        return 0

    if maximizing:
        best = -inf
        for r, c in _ordered_moves(board):
            board[r][c] = ai_sym
            score = _minimax(board, depth + 1, False, ai_sym, human_sym, alpha, beta)
            board[r][c] = EMPTY
            best = max(best, score)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return int(best)
    else:
        best = inf
        for r, c in _ordered_moves(board):
            board[r][c] = human_sym
            score = _minimax(board, depth + 1, True, ai_sym, human_sym, alpha, beta)
            board[r][c] = EMPTY
            best = min(best, score)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return int(best)


# ---------------------------------------------------------------------------
# Move selector
# ---------------------------------------------------------------------------

def _can_win_in_one(board: Board, sym: str) -> Optional[Tuple[int, int]]:
    for r, c in _ordered_moves(board):
        board[r][c] = sym
        if check_winner(board):
            board[r][c] = EMPTY
            return (r, c)
        board[r][c] = EMPTY
    return None


def find_best_move(
    board:    Board,
    ai_sym:   str = YELLOW,
    human_sym: str = RED,
) -> Tuple[int, int, int, str]:
    """
    Return (row, col, minimax_score, status_message).
    status options:
      "Winning move available"
      "Blocking move required"
      "Forced draw"
      "Playing"
    Returns (-1, -1, 0, <reason>) if no move possible.
    """
    moves = _ordered_moves(board)
    if not moves:
        return -1, -1, 0, "No legal moves"

    best_score = -inf
    best_move  = moves[0]

    for r, c in moves:
        board[r][c] = ai_sym
        score = _minimax(board, 0, False, ai_sym, human_sym, -inf, inf)
        board[r][c] = EMPTY
        if score > best_score:
            best_score = score
            best_move  = (r, c)

    br, bc = best_move

    # Classify status
    if best_score > 0:
        status = "Winning move available"
    elif best_score < 0:
        # AI is losing; check if it at least blocks
        if _can_win_in_one(board, human_sym):
            status = "Blocking move required"
        else:
            status = "Playing"
    else:
        # score == 0 → draw with optimal play
        status = "Forced draw"

    return br, bc, int(best_score), status


# ---------------------------------------------------------------------------
# TicTacToeGame
# ---------------------------------------------------------------------------

class TicTacToeGame:
    """
    Central game state using E/R/Y symbols throughout.

    user_sym: symbol for the human player  (default RED  = 'R')
    ai_sym:   symbol for the robot player  (default YELLOW = 'Y')
    """

    def __init__(self, user_sym: str = RED, ai_sym: str = YELLOW) -> None:
        self.user_sym  = user_sym
        self.ai_sym    = ai_sym

        self.board: Board = [[EMPTY] * 3 for _ in range(3)]
        self.current_player  = user_sym   # user always moves first
        self.game_over       = False
        self.winner: Optional[str] = None
        self.winning_cells: List[Tuple[int, int]] = []
        self.game_status_msg = "Waiting for move"

        # For win-message display timing
        self.game_just_ended = False
        self.win_message_time = 0.0

        # Logging: only print when board changes
        self._last_logged_board: Board = [[EMPTY] * 3 for _ in range(3)]

        # Pending suggested move from Minimax
        self.suggested_move: Optional[Tuple[int, int]] = None
        self.suggested_status = ""
        self.suggested_score  = 0
        self._last_suggestion_board: Optional[tuple] = None

    # ------------------------------------------------------------------
    # Board helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.board           = [[EMPTY] * 3 for _ in range(3)]
        self.current_player  = self.user_sym
        self.game_over       = False
        self.winner          = None
        self.winning_cells   = []
        self.game_just_ended = False
        self.win_message_time = 0.0
        self.game_status_msg = "Waiting for move"
        self.suggested_move  = None
        self.suggested_status = ""
        self.suggested_score  = 0
        self._last_suggestion_board = None
        self._last_logged_board = [[EMPTY] * 3 for _ in range(3)]
        comms.send_ui_update("reset")

    def set_board(self, board: Board) -> None:
        """Replace the internal board (used when syncing from YOLO detections)."""
        self.board = [row[:] for row in board]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> Tuple[bool, str]:
        return validate_board(self.board, self.user_sym, self.ai_sym)

    # ------------------------------------------------------------------
    # Win / draw checking
    # ------------------------------------------------------------------

    def _find_winning_cells(self, winner_sym: str) -> List[Tuple[int, int]]:
        b = self.board
        for r in range(3):
            if b[r][0] == b[r][1] == b[r][2] == winner_sym:
                return [(r, 0), (r, 1), (r, 2)]
        for c in range(3):
            if b[0][c] == b[1][c] == b[2][c] == winner_sym:
                return [(0, c), (1, c), (2, c)]
        if b[0][0] == b[1][1] == b[2][2] == winner_sym:
            return [(0, 0), (1, 1), (2, 2)]
        if b[0][2] == b[1][1] == b[2][0] == winner_sym:
            return [(0, 2), (1, 1), (2, 0)]
        return []

    def check_game_end(self) -> bool:
        """Check for terminal state. Returns True if game just ended."""
        if self.game_over:
            return False
        winner = check_winner(self.board)
        if winner is not None:
            self.winner          = winner
            self.game_over       = True
            self.game_just_ended = True
            self.win_message_time = time.time()
            self.winning_cells   = self._find_winning_cells(winner)
            self.game_status_msg = (
                "Win" if winner == self.ai_sym else "Loss"
            )
            ui_sym = "O" if winner == self.ai_sym else "X"
            comms.send_ui_update(f"win {ui_sym}")
            return True
        if is_full(self.board):
            self.winner          = None
            self.game_over       = True
            self.game_just_ended = True
            self.win_message_time = time.time()
            self.game_status_msg = "Draw"
            comms.send_ui_update("tie")
            return True
        return False

    # ------------------------------------------------------------------
    # Minimax suggestion  (always computed for current board state)
    # ------------------------------------------------------------------

    def update_suggestion(self) -> None:
        """
        Re-run Minimax if the board has changed.
        Sets self.suggested_move, self.suggested_status, self.suggested_score.
        Skips if game is over or board is invalid.
        """
        sig = tuple(tuple(row) for row in self.board)
        if sig == self._last_suggestion_board:
            return
        self._last_suggestion_board = sig

        # Skip if game is over
        if self.game_over:
            self.suggested_move   = None
            self.suggested_status = self.game_status_msg
            self.suggested_score  = 0
            return

        # Validate
        valid, reason = self.validate()
        if not valid:
            self.suggested_move   = None
            self.suggested_status = f"Invalid board: {reason}"
            self.suggested_score  = 0
            if os.environ.get("TICTACTOE_DEBUG"):
                print(f"[MINIMAX] Skipped — {reason}")
            return

        # Already won / drawn?
        winner = check_winner(self.board)
        if winner is not None:
            self.suggested_move   = None
            self.suggested_status = "Win" if winner == self.ai_sym else "Loss"
            self.suggested_score  = 0
            return
        if is_draw(self.board):
            self.suggested_move   = None
            self.suggested_status = "Draw"
            self.suggested_score  = 0
            return

        # Run Minimax
        board_copy = [row[:] for row in self.board]
        row, col, score, status = find_best_move(board_copy, self.ai_sym, self.user_sym)
        self.suggested_move   = (row, col) if row != -1 else None
        self.suggested_status = status
        self.suggested_score  = score
        self.game_status_msg  = status

        if os.environ.get("TICTACTOE_DEBUG"):
            counts = board_counts(self.board)
            print(
                f"[MINIMAX] board_R={counts[RED]} board_Y={counts[YELLOW]} "
                f"→ move=({row},{col}) score={score} status={status}"
            )

    # ------------------------------------------------------------------
    # Move application (robot arm)
    # ------------------------------------------------------------------

    def apply_ai_move(self) -> bool:
        """
        Apply the current suggested_move to the board and send robot command.
        Returns True on success.
        """
        if self.suggested_move is None or self.game_over:
            return False
        row, col = self.suggested_move
        if self.board[row][col] != EMPTY:
            return False

        self.board[row][col] = self.ai_sym

        # Robot arm command: map (row,col) → legacy ID 1–9
        map_grid = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        move_id  = map_grid[row][col]
        comms.send(str(move_id))
        comms.send_ui_update(f"move {row} {col} O")

        self.current_player = self.user_sym
        self.check_game_end()
        return True

    # ------------------------------------------------------------------
    # Sync board from YOLO  (detect new user move, detect AI confirmation)
    # ------------------------------------------------------------------

    def sync_from_detection(self, detected_board: Board) -> Optional[str]:
        """
        Compare detected board against internal board.
        Returns event string: 'user_move', 'ai_confirmed', 'no_change', 'invalid'.
        Does NOT auto-apply; caller decides what to do.
        """
        valid, reason = validate_board(detected_board, self.user_sym, self.ai_sym)
        if not valid:
            return "invalid"

        # Count new user pieces
        new_user = []
        new_ai   = []
        missing_ai = []
        for r in range(3):
            for c in range(3):
                prev = self.board[r][c]
                curr = detected_board[r][c]
                if prev == EMPTY and curr == self.user_sym:
                    new_user.append((r, c))
                elif prev == EMPTY and curr == self.ai_sym:
                    new_ai.append((r, c))
                elif prev == self.ai_sym and curr == EMPTY:
                    missing_ai.append((r, c))

        if len(new_user) == 1 and len(new_ai) == 0 and len(missing_ai) == 0:
            return "user_move"
        if len(new_ai) >= 1 and len(new_user) == 0:
            return "ai_confirmed"
        if new_user == [] and new_ai == [] and missing_ai == []:
            return "no_change"
        return "invalid"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_status(self) -> None:
        """Print board only when it has changed."""
        if self.board == self._last_logged_board:
            return
        self._last_logged_board = [row[:] for row in self.board]
        counts = board_counts(self.board)
        turn   = "User" if self.current_player == self.user_sym else "Robot"
        print(f"\n{'='*44}")
        print(f"  TIC-TAC-TOE  |  Turn: {turn}  |  {self.game_status_msg}")
        print(f"{'='*44}")
        for i, row in enumerate(self.board):
            def sym(s):
                return ("X" if s == self.user_sym else ("O" if s == self.ai_sym else " "))
            print(f"  {sym(row[0])} | {sym(row[1])} | {sym(row[2])}")
            if i < 2:
                print("  ----------")
        print(f"{'='*44}")
        if self.game_over:
            if self.winner:
                w = "Robot" if self.winner == self.ai_sym else "User"
                print(f"  🎉 GAME OVER — {w} wins!")
            else:
                print("  🤝 GAME OVER — Draw!")
        elif self.suggested_move:
            r, c = self.suggested_move
            print(f"  🤖 AI suggests: row={r} col={c}  [{self.suggested_status}]")
        print(f"{'='*44}\n")
