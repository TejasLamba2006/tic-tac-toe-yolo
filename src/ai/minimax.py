from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Sequence


Board = list[list[str]]


@dataclass(frozen=True)
class MoveRecommendation:
    """Best move returned by the minimax search."""

    row: int
    col: int
    score: int
    ai_player: str
    human_player: str


def opponent(player: str) -> str:
    if player == "R":
        return "Y"
    if player == "Y":
        return "R"
    raise ValueError(f"Unsupported player token: {player}")


def check_winner(board: Sequence[Sequence[str]]) -> str | None:
    """Return the winning player symbol or ``None`` if the game is still open."""

    lines = []
    lines.extend(board)
    lines.extend([[board[row][col] for row in range(3)] for col in range(3)])
    lines.append([board[i][i] for i in range(3)])
    lines.append([board[i][2 - i] for i in range(3)])

    for symbol in ("R", "Y"):
        if any(all(cell == symbol for cell in line) for line in lines):
            return symbol
    return None


def is_draw(board: Sequence[Sequence[str]]) -> bool:
    return check_winner(board) is None and all(cell != "E" for row in board for cell in row)


def _ordered_moves(board: Sequence[Sequence[str]]):
    preferred = [
        (1, 1),
        (0, 0),
        (0, 2),
        (2, 0),
        (2, 2),
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
    ]
    yield from [move for move in preferred if board[move[0]][move[1]] == "E"]


def minimax(
    board: Board,
    depth: int,
    maximizing: bool,
    ai_player: str,
    human_player: str,
    alpha: float = -inf,
    beta: float = inf,
) -> int:
    """Return the minimax score from the point of view of ``ai_player``."""

    winner = check_winner(board)
    if winner == ai_player:
        return 10 - depth
    if winner == human_player:
        return depth - 10
    if is_draw(board):
        return 0

    if maximizing:
        best_score = -inf
        for row, col in _ordered_moves(board):
            board[row][col] = ai_player
            score = minimax(board, depth + 1, False, ai_player,
                            human_player, alpha, beta)
            board[row][col] = "E"
            if score > best_score:
                best_score = score
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return int(best_score)

    best_score = inf
    for row, col in _ordered_moves(board):
        board[row][col] = human_player
        score = minimax(board, depth + 1, True, ai_player,
                        human_player, alpha, beta)
        board[row][col] = "E"
        if score < best_score:
            best_score = score
        beta = min(beta, best_score)
        if beta <= alpha:
            break
    return int(best_score)


def find_best_move(board: Board, ai_player: str = "Y") -> MoveRecommendation | None:
    """Return the optimal move for ``ai_player`` or ``None`` if the game is over."""

    if check_winner(board) is not None or is_draw(board):
        return None

    human_player = opponent(ai_player)
    best_score = -inf
    best_move: tuple[int, int] | None = None

    for row, col in _ordered_moves(board):
        board[row][col] = ai_player
        score = minimax(board, 0, False, ai_player, human_player)
        board[row][col] = "E"

        if score > best_score or (score == best_score and best_move is not None and (row, col) < best_move):
            best_score = score
            best_move = (row, col)

    if best_move is None:
        return None

    return MoveRecommendation(
        row=best_move[0],
        col=best_move[1],
        score=int(best_score),
        ai_player=ai_player,
        human_player=human_player,
    )
