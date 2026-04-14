"""
Training script for Tabular Q-Learning.

Trains PLAYER1 against a random opponent and plots two things in real time:
  - Rolling win rate (over the last WINDOW episodes)
  - Q-table size (number of unique (state, action) pairs seen)

The Q-table size chart is the interesting one — it shows the state-space
explosion. It will keep growing because the agent visits new states every
game and can never generalize. This is the wall that kills tabular Q-learning.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from rl_connect_four.mdp import ConnectFourMDP
from rl_connect_four.engine import PLAYER1
from rl_connect_four.tabular_q import TabularQAgent, train

# --- Config ---
N_EPISODES    = 50_000
WINDOW        = 1_000   # rolling win rate window
UPDATE_EVERY  = 500     # redraw graph every N episodes

# --- Setup ---
mdp   = ConnectFourMDP()
agent = TabularQAgent(player=PLAYER1, alpha=0.1, gamma=0.99, epsilon=1.0)

# Storage for plotting
all_rewards    = []
win_rates      = []
qtable_sizes   = []
x_axis         = []

# --- Live plot setup ---
plt.ion()
fig = plt.figure(figsize=(12, 5))
gs  = gridspec.GridSpec(1, 2, figure=fig)

ax_wr = fig.add_subplot(gs[0])
ax_qt = fig.add_subplot(gs[1])

ax_wr.set_title("Rolling Win Rate")
ax_wr.set_xlabel("Episode")
ax_wr.set_ylabel(f"Win rate (last {WINDOW} episodes)")
ax_wr.set_ylim(0, 1)

ax_qt.set_title("Q-Table Size (State Space Explosion)")
ax_qt.set_xlabel("Episode")
ax_qt.set_ylabel("Unique (state, action) pairs")

line_wr, = ax_wr.plot([], [], color="steelblue")
line_qt, = ax_qt.plot([], [], color="tomato")

plt.tight_layout()
plt.show()


def update_plot():
    line_wr.set_data(x_axis, win_rates)
    line_qt.set_data(x_axis, qtable_sizes)
    ax_wr.relim(); ax_wr.autoscale_view()
    ax_qt.relim(); ax_qt.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


# --- Training loop with periodic plot updates ---
import random
from rl_connect_four.engine import other_player

print(f"Training for {N_EPISODES:,} episodes...")

for ep in range(N_EPISODES):
    # Run one episode manually so we can update the plot mid-training
    state = mdp.initial_state()

    while not mdp.is_terminal(state):
        legal = mdp.get_actions(state)

        if state.current_player == agent.player:
            action = agent.select_action(state, legal)
        else:
            action = random.choice(legal)

        next_state = mdp.step(state, action)
        done       = mdp.is_terminal(next_state)

        if state.current_player == agent.player:
            reward     = mdp.get_reward(next_state, agent.player) if done else 0.0
            next_legal = mdp.get_actions(next_state) if not done else []
            agent.update(state, action, reward, next_state, next_legal, done)

        state = next_state

    all_rewards.append(mdp.get_reward(state, agent.player))

    # Decay epsilon
    agent.epsilon = max(0.05, agent.epsilon * 0.9995)

    # Periodic update
    if (ep + 1) % UPDATE_EVERY == 0:
        window_rewards = all_rewards[-WINDOW:]
        win_rate = sum(1 for r in window_rewards if r == 1.0) / len(window_rewards)

        win_rates.append(win_rate)
        qtable_sizes.append(len(agent.q_table))
        x_axis.append(ep + 1)

        update_plot()
        print(
            f"Episode {ep+1:>6,} | "
            f"Win rate: {win_rate:.1%} | "
            f"Q-table: {len(agent.q_table):>8,} entries | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

print("\nDone. Close the plot window to exit.")
plt.ioff()
plt.show()
