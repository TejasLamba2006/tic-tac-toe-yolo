from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .minimax import MoveRecommendation, check_winner, find_best_move, is_draw


@dataclass(frozen=True)
class MoveDecision:
    """High-level recommendation plus a status string for the UI."""

    recommendation: MoveRecommendation | None
    winner: str | None
    draw: bool
    valid: bool
    message: str


def _symbol_counts(board: Sequence[Sequence[str]]) -> dict[str, int]:
    counts = {"E": 0, "R": 0, "Y": 0}
    for row in board:
        for cell in row:
            counts[cell] = counts.get(cell, 0) + 1
    return counts


def validate_board(board: Sequence[Sequence[str]]) -> bool:
    """Return ``True`` when the board only contains supported symbols and the counts look sane."""

    counts = _symbol_counts(board)
    if any(symbol not in {"E", "R", "Y"} for row in board for symbol in row):
        return False
    return abs(counts["R"] - counts["Y"]) <= 1


def recommend_move(board: Sequence[Sequence[str]], ai_player: str = "Y") -> MoveDecision:
    """Return the best move together with a human-readable status message."""

    winner = check_winner(board)
    draw = is_draw(board)
    valid = validate_board(board)

    if winner is not None:
        return MoveDecision(
            recommendation=None,
            winner=winner,
            draw=False,
            valid=valid,
            message=f"Game over. {winner} already wins.",
        )

    if draw:
        return MoveDecision(
            recommendation=None,
            winner=None,
            draw=True,
            valid=valid,
            message="Game over. The board is full and the result is a draw.",
        )

    recommendation = find_best_move([list(row)
                                    for row in board], ai_player=ai_player)
    if recommendation is None:
        return MoveDecision(
            recommendation=None,
            winner=None,
            draw=False,
            valid=valid,
            message="No legal move was found.",
        )

    if not valid:
        message = "Board state looks inconsistent, but a best-effort move was computed."
    else:
        message = f"Best move for {ai_player}: row={recommendation.row}, col={recommendation.col}."

    return MoveDecision(
        recommendation=recommendation,
        winner=None,
        draw=False,
        valid=valid,
        message=message,
    )
