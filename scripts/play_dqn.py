"""
Interactive match: you (PLAYER2 / O) vs the trained DQN agent (PLAYER1 / X).
Loads weights from outputs/dqn_weights.pt.
"""

import torch
from rl_connect_four.engine import PLAYER1, PLAYER2, print_board, check_win, is_draw
from rl_connect_four.mdp import ConnectFourMDP
from rl_connect_four.dqn import DQNAgent, QNetwork, encode_state

WEIGHTS_PATH = "outputs/dqn_weights.pt"


def load_agent() -> DQNAgent:
    agent = DQNAgent()
    agent.online.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))
    agent.online.eval()
    agent.epsilon = 0.0  # always greedy
    return agent


def get_human_move(valid: list[int]) -> int:
    while True:
        try:
            col = int(input(f"Your move {valid}: "))
            if col in valid:
                return col
            print("Invalid column, try again.")
        except ValueError:
            print("Enter a number.")


def play():
    mdp   = ConnectFourMDP()
    agent = load_agent()
    state = mdp.initial_state()

    choice = input("Do you want to go first? (y/n): ").strip().lower()
    human_first = choice == "y"
    human_player = PLAYER1 if human_first else PLAYER2
    agent_player = PLAYER2 if human_first else PLAYER1
    print(f"\nYou are {'X' if human_first else 'O'}. Agent is {'O' if human_first else 'X'}.\n")
    print_board(state.board)

    # Alternate turns; current_player tracks whose turn it is via state
    while True:
        current = state.current_player

        if current == human_player:
            valid  = mdp.get_actions(state)
            action = get_human_move(valid)
        else:
            valid  = mdp.get_actions(state)
            action = agent.select_action(encode_state(state), valid)
            print(f"Agent plays column {action}")

        state = mdp.step(state, action)
        print_board(state.board)

        if check_win(state.board, current):
            print("You win!" if current == human_player else "Agent wins!")
            break
        if is_draw(state.board):
            print("Draw!")
            break


if __name__ == "__main__":
    play()
