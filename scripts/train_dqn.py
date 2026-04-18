"""
Train a DQN agent (PLAYER1) to play Connect Four against a random opponent (PLAYER2).                              
Saves the trained model weights and plots a win-rate curve.

Only PLAYER1's transitions are stored. After PLAYER1 acts, we let
PLAYER2 respond immediately and store (s, a, r, s'') where s'' is
the resulting state after PLAYER2's move — always PLAYER1's turn.
This keeps the replay buffer consistent: all states are encoded from
the acting player's perspective at the start of their turn.
"""

import random
import os
import torch
import matplotlib.pyplot as plt

from rl_connect_four.mdp import ConnectFourMDP, State
from rl_connect_four.engine import PLAYER1, PLAYER2
from rl_connect_four.dqn import DQNAgent, encode_state

NUM_EPISODES    = 25_000
LOG_EVERY       = 500
EVAL_EVERY      = 500
EVAL_EPISODES   = 200


def random_action(mdp: ConnectFourMDP, state: State) -> int:
    return random.choice(mdp.get_actions(state))

def run_episode(agent: DQNAgent, mdp: ConnectFourMDP) -> float:
    """
    Play one game. Returns PLAYER1's total reward for the episode.
    PLAYER2 acts randomly; its transitions are never stored.
    """

    state = mdp.initial_state()
    total_reward = 0.0

    while True:
        # ---- PLAYER1's Turn ----
        s_enc       = encode_state(state)
        valid       = mdp.get_actions(state)
        action      = agent.select_action(s_enc, valid)
        next_state  = mdp.step(state, action)

        if mdp.is_terminal(next_state):
            # Game ended on PLAYER1's move (win or draw)
            reward = mdp.get_reward(next_state, perspective=PLAYER1)
            agent.store(s_enc, action, reward, encode_state(next_state), valid_next_actions=[], done=True)
            agent.update()
            total_reward += reward
            break

        # ---- PLAYER2's Turn ----
        p2_action = random_action(mdp, next_state)
        after_p2 = mdp.step(next_state, p2_action)

        if mdp.is_terminal(after_p2):
            # Game ended on PLAYER2's move
            reward = mdp.get_reward(after_p2, perspective=PLAYER1)
            agent.store(s_enc, action, reward, encode_state(after_p2), valid_next_actions=[], done=True)
            agent.update()
            total_reward += reward
            break

        # Mid-Game
        agent.store(s_enc, action, 0.0, encode_state(after_p2), valid_next_actions=mdp.get_actions(after_p2), done=False)
        agent.update()
        state = after_p2
    
    return total_reward


def eval_agent(agent: DQNAgent, mdp: ConnectFourMDP, n: int = EVAL_EPISODES) -> float:
    """Run n greedy games (ε=0) and return win rate. No learning, no replay."""
    saved_eps = agent.epsilon
    agent.epsilon = 0.0
    wins = 0
    for _ in range(n):
        state = mdp.initial_state()
        while True:
            valid = mdp.get_actions(state)
            action = agent.select_action(encode_state(state), valid)
            state = mdp.step(state, action)
            if mdp.is_terminal(state):
                if mdp.get_reward(state, perspective=PLAYER1) > 0:
                    wins += 1
                break
            state = mdp.step(state, random_action(mdp, state))
            if mdp.is_terminal(state):
                break
    agent.epsilon = saved_eps
    return wins / n


def setup_live_plot():
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("DQN Win Rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    train_line, = ax.plot([], [], color="steelblue", label="train (ε-greedy)")
    eval_line,  = ax.plot([], [], color="tomato",    label="eval (greedy)")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig, ax, train_line, eval_line


def update_plot(fig, ax, train_line, eval_line, x, win_rates, eval_x, eval_rates):
    train_line.set_data(x, win_rates)
    eval_line.set_data(eval_x, eval_rates)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


def train():
    mdp   = ConnectFourMDP()
    agent = DQNAgent()

    fig, ax, train_line, eval_line = setup_live_plot()

    recent_rewards = []
    x_axis, win_rates = [], []
    eval_x, eval_rates = [], []

    for episode in range(1, NUM_EPISODES + 1):
        reward = run_episode(agent, mdp)
        agent.decay_epsilon()
        recent_rewards.append(reward)

        if episode % LOG_EVERY == 0:
            window   = recent_rewards[-LOG_EVERY:]
            win_rate = sum(r > 0 for r in window) / len(window)
            x_axis.append(episode)
            win_rates.append(win_rate)

        if episode % EVAL_EVERY == 0:
            eval_rate = eval_agent(agent, mdp)
            eval_x.append(episode)
            eval_rates.append(eval_rate)
            print(f"ep {episode:>6} | train: {win_rates[-1]:.2%} | eval: {eval_rate:.2%} | ε: {agent.epsilon:.4f}")
            update_plot(fig, ax, train_line, eval_line, x_axis, win_rates, eval_x, eval_rates)

    os.makedirs("outputs", exist_ok=True)
    plt.ioff()
    plt.savefig("outputs/dqn_win_rate.png", dpi=150, bbox_inches="tight")
    torch.save(agent.online.state_dict(), "outputs/dqn_weights.pt")
    print("\nSaved plot → outputs/dqn_win_rate.png")
    print("Saved weights → outputs/dqn_weights.pt")
    print("Done. Close the plot window to exit.")
    plt.show()


if __name__ == "__main__":
    train()
