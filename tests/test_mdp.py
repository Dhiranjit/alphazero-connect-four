import pytest
from rl_connect_four.mdp import ConnectFourMDP
from rl_connect_four.engine import PLAYER1, PLAYER2, ROWS, COLS, EMPTY


@pytest.fixture
def mdp():
    return ConnectFourMDP()


@pytest.fixture
def initial(mdp):
    return mdp.initial_state()


# --- initial_state ---

def test_initial_board_is_empty(mdp, initial):
    assert all(cell == EMPTY for row in initial.board for cell in row)

def test_initial_player_is_player1(mdp, initial):
    assert initial.current_player == PLAYER1

def test_initial_board_dimensions(mdp, initial):
    assert len(initial.board) == ROWS
    assert all(len(row) == COLS for row in initial.board)


# --- get_actions ---

def test_all_columns_legal_on_empty_board(mdp, initial):
    assert mdp.get_actions(initial) == list(range(COLS))

def test_full_column_not_legal(mdp, initial):
    # Fill column 0 completely
    state = initial
    for _ in range(ROWS):
        state = mdp.step(state, 0)
    assert 0 not in mdp.get_actions(state)


# --- step ---

def test_step_piece_lands_at_bottom(mdp, initial):
    state = mdp.step(initial, 0)
    # Bottom row, first column should have PLAYER1's piece
    assert state.board[ROWS - 1][0] == PLAYER1

def test_step_piece_stacks(mdp, initial):
    state = mdp.step(initial, 0)  # PLAYER1 drops col 0
    state = mdp.step(state, 0)    # PLAYER2 drops col 0
    assert state.board[ROWS - 1][0] == PLAYER1
    assert state.board[ROWS - 2][0] == PLAYER2

def test_step_turn_flips(mdp, initial):
    state = mdp.step(initial, 0)
    assert state.current_player == PLAYER2
    state = mdp.step(state, 0)
    assert state.current_player == PLAYER1

def test_step_returns_new_state(mdp, initial):
    # State is immutable — step must return a new object
    state = mdp.step(initial, 0)
    assert state is not initial


# --- is_terminal ---

def test_not_terminal_on_empty_board(mdp, initial):
    assert mdp.is_terminal(initial) is False

def test_terminal_on_horizontal_win(mdp, initial):
    # PLAYER1 wins across the bottom row (cols 0-3)
    # Alternate: P1 col0, P2 col4, P1 col1, P2 col5, P1 col2, P2 col6, P1 col3
    state = initial
    for p1_col, p2_col in [(0, 4), (1, 5), (2, 6)]:
        state = mdp.step(state, p1_col)
        state = mdp.step(state, p2_col)
    state = mdp.step(state, 3)  # PLAYER1 wins
    assert mdp.is_terminal(state) is True

def test_terminal_on_draw(mdp, initial):
    # Fill board without anyone winning
    # Column fill order chosen to avoid any 4-in-a-row
    state = initial
    # Alternate filling columns in an order that avoids a win
    fill_order = [0, 2, 4, 6, 1, 3, 5,
                  0, 2, 4, 6, 1, 3, 5,
                  0, 2, 4, 6, 1, 3, 5,
                  0, 2, 4, 6, 1, 3, 5,
                  0, 2, 4, 6, 1, 3, 5,
                  0, 2, 4, 6, 1, 3, 5]
    for col in fill_order:
        if not mdp.is_terminal(state):
            state = mdp.step(state, col)
    # Board full or terminal
    assert mdp.is_terminal(state) is True


# --- get_reward ---

def test_reward_winner_gets_plus_one(mdp, initial):
    state = initial
    for p1_col, p2_col in [(0, 4), (1, 5), (2, 6)]:
        state = mdp.step(state, p1_col)
        state = mdp.step(state, p2_col)
    state = mdp.step(state, 3)  # PLAYER1 wins
    assert mdp.get_reward(state, PLAYER1) == 1.0

def test_reward_loser_gets_minus_one(mdp, initial):
    state = initial
    for p1_col, p2_col in [(0, 4), (1, 5), (2, 6)]:
        state = mdp.step(state, p1_col)
        state = mdp.step(state, p2_col)
    state = mdp.step(state, 3)  # PLAYER1 wins
    assert mdp.get_reward(state, PLAYER2) == -1.0

def test_reward_draw_is_zero(mdp, initial):
    state = initial
    fill_order = [0, 2, 4, 6, 1, 3, 5,
                  0, 2, 4, 6, 1, 3, 5,
                  0, 2, 4, 6, 1, 3, 5,
                  0, 2, 4, 6, 1, 3, 5,
                  0, 2, 4, 6, 1, 3, 5,
                  0, 2, 4, 6, 1, 3, 5]
    for col in fill_order:
        if not mdp.is_terminal(state):
            state = mdp.step(state, col)
    # Only check reward if it ended in a draw (no winner)
    from rl_connect_four.engine import check_win
    if not check_win(state.board, PLAYER1) and not check_win(state.board, PLAYER2):
        assert mdp.get_reward(state, PLAYER1) == 0.0
        assert mdp.get_reward(state, PLAYER2) == 0.0
