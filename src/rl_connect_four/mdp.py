from dataclasses import dataclass
from rl_connect_four.engine import (
    create_board, drop_piece, get_legal_actions,
    check_win, is_draw, other_player, PLAYER1
)


@dataclass(frozen=True)
class State:
    """             
    Immutable snapshot of the game at a point in time.
                                                        
    Using a frozen dataclass with a tuple board means States are hashable —
    MCTS can use them as dictionary keys for its tree nodes.                                      
    """
    board: tuple          # tuple-of-tuples so it's immutable and hashable
    current_player: int   # the player whose turn it is to move


class ConnectFourMDP:
    """
    Formalises Connect Four as a Markov Decision Process.

    Every RL algorithm (Q-Learning, DQN, policy gradient, ... AlphaZero) talks
    to the game exclusively through this interface.

    MDP components:
      - State space     : State(board, current_player)
      - Action space    : get_actions(state)
      - Transition      : step(state, action) → next state  [deterministic]
      - Reward function : get_reward(state, perspective)
      - Discount factor : not here — belongs to the learning algorithm

    Convention used throughout:
      - 'current_player'  = the player who is ABOUT TO move next
      - 'previous_player' = the player who JUST moved (i.e. other_player(current))
      - Rewards are always from the perspective of a given player (+1 win,
        -1 loss, 0 draw).
    """

    def initial_state(self) -> State:
        """Return the empty board with PLAYER1 to move first."""
        board = tuple(tuple(row) for row in create_board())
        return State(board=board, current_player=PLAYER1)
    
    def get_actions(self, state: State) -> list[int]:
        """Return the list of legal column indices in the given state"""
        return get_legal_actions(state.board)
    
    def step(self, state: State, action: int) -> State:
        """
        Apply `action` (a column index) for state.current_player and return
        the resulting state where it is the OTHER player's turn.           
        """  
        # engine.drop_piece() works only with list-of-list;
        list_board = [list(row) for row in state.board]
        new_list_board = drop_piece(board=list_board, col=action, player=state.current_player)
        new_board = tuple(tuple(row) for row in new_list_board)

        return State(board=new_board, 
                     current_player=other_player(state.current_player))
    
    def is_terminal(self, state: State) -> bool:
        """
        Return True if the game is over.

        A state is teminal when the PREVIOUS player's move caused a win, or
        the board is completely full (draw).
        """
        previous_player = other_player(state.current_player)
        return check_win(state.board, previous_player) or is_draw(state.board)
    
    def get_reward(self, state: State, perspective: int) -> float:
        """
        Return the reward for a TERMINAL state from 'perspective's point of view.

          +1.0  — perspective won
          -1.0  — perspective lost
           0.0  — draw

        Calling this on a non-terminal state is undefined; callers must check
        is_terminal() first.
        """
        previous_player = other_player(state.current_player)

        if check_win(state.board, previous_player):
            return 1.0 if previous_player == perspective else -1.0
        return 0.0 # draw
    



