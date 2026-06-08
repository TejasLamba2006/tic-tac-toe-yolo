from __future__ import annotations

from src.ai.minimax import check_winner, find_best_move, is_draw, minimax


def test_check_winner_detects_rows_columns_and_diagonals() -> None:
    row_board = [
        ["R", "R", "R"],
        ["E", "Y", "E"],
        ["Y", "E", "E"],
    ]
    column_board = [
        ["Y", "R", "E"],
        ["Y", "R", "E"],
        ["E", "R", "E"],
    ]
    diagonal_board = [
        ["Y", "R", "E"],
        ["R", "Y", "E"],
        ["E", "R", "Y"],
    ]

    assert check_winner(row_board) == "R"
    assert check_winner(column_board) == "R"
    assert check_winner(diagonal_board) == "Y"


def test_draw_detection() -> None:
    board = [
        ["R", "Y", "R"],
        ["Y", "R", "Y"],
        ["Y", "R", "Y"],
    ]

    assert is_draw(board) is True


def test_minimax_blocks_immediate_loss() -> None:
    board = [
        ["R", "R", "E"],
        ["Y", "E", "E"],
        ["E", "Y", "E"],
    ]

    move = find_best_move(board, ai_player="Y")
    assert move is not None
    assert (move.row, move.col) == (0, 2)


def test_minimax_prefers_winning_move() -> None:
    board = [
        ["Y", "Y", "E"],
        ["R", "R", "E"],
        ["E", "E", "E"],
    ]

    move = find_best_move(board, ai_player="Y")
    assert move is not None
    assert (move.row, move.col) == (0, 2)
    assert minimax(board, 0, True, "Y", "R") >= 0
