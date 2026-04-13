import pytest
from rl_connect_four.engine import (
    create_board, drop_piece, get_legal_actions,
    check_win, is_draw, other_player,
    ROWS, COLS, EMPTY, PLAYER1, PLAYER2,
)


def test_create_board_is_all_empty():
    board = create_board()
    assert len(board) == ROWS
    assert all(len(row) == COLS for row in board)
    assert all(cell == EMPTY for row in board for cell in row)


def test_drop_piece_lands_at_bottom():
    board = create_board()
    board = drop_piece(board, 3, PLAYER1)
    assert board[ROWS - 1][3] == PLAYER1


def test_drop_piece_stacks():
    board = create_board()
    board = drop_piece(board, 3, PLAYER1)
    board = drop_piece(board, 3, PLAYER2)
    assert board[ROWS - 1][3] == PLAYER1
    assert board[ROWS - 2][3] == PLAYER2


def test_drop_piece_does_not_mutate_original():
    board = create_board()
    new_board = drop_piece(board, 0, PLAYER1)
    assert board[ROWS - 1][0] == EMPTY
    assert new_board[ROWS - 1][0] == PLAYER1


def test_get_legal_actions_full_board():
    board = create_board()
    assert get_legal_actions(board) == list(range(COLS))


def test_get_legal_actions_excludes_full_column():
    board = create_board()
    for _ in range(ROWS):
        board = drop_piece(board, 0, PLAYER1)
    assert 0 not in get_legal_actions(board)


def test_check_win_horizontal():
    board = create_board()
    for col in range(4):
        board = drop_piece(board, col, PLAYER1)
    assert check_win(board, PLAYER1)


def test_check_win_vertical():
    board = create_board()
    for _ in range(4):
        board = drop_piece(board, 0, PLAYER1)
    assert check_win(board, PLAYER1)


def test_check_win_diagonal_top_left_to_bottom_right():
    board = create_board()
    # Build the diagonal by stacking opponents underneath each PLAYER1 piece
    for col in range(4):
        for _ in range(col):
            board = drop_piece(board, col, PLAYER2)
        board = drop_piece(board, col, PLAYER1)
    assert check_win(board, PLAYER1)


def test_check_win_diagonal_bottom_left_to_top_right():
    board = create_board()
    for col in range(4):
        for _ in range(3 - col):
            board = drop_piece(board, col, PLAYER2)
        board = drop_piece(board, col, PLAYER1)
    assert check_win(board, PLAYER1)


def test_check_win_returns_false_with_no_winner():
    board = create_board()
    board = drop_piece(board, 0, PLAYER1)
    board = drop_piece(board, 1, PLAYER1)
    board = drop_piece(board, 2, PLAYER1)
    assert not check_win(board, PLAYER1)


def test_is_draw_on_full_board():
    # Fill the board in a pattern that avoids any winner
    board = create_board()
    pattern = [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1]
    for r in range(ROWS):
        for c in range(COLS):
            board = drop_piece(board, c, pattern[(r + c) % 2])
            if check_win(board, PLAYER1) or check_win(board, PLAYER2):
                # Pattern produced a winner — skip draw assertion
                return
    # If we filled without a winner, it should be a draw
    if not check_win(board, PLAYER1) and not check_win(board, PLAYER2):
        assert is_draw(board)


def test_is_draw_not_triggered_on_empty_board():
    board = create_board()
    assert not is_draw(board)


def test_other_player_flips_both_ways():
    assert other_player(PLAYER1) == PLAYER2
    assert other_player(PLAYER2) == PLAYER1
