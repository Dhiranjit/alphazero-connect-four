import pytest
import random
from rl_connect_four.mdp import ConnectFourMDP, State
from rl_connect_four.engine import PLAYER1, PLAYER2
from rl_connect_four.tabular_q import TabularQAgent, train


@pytest.fixture
def mdp():
    return ConnectFourMDP()


@pytest.fixture
def agent():
    return TabularQAgent(player=PLAYER1)


@pytest.fixture
def initial(mdp):
    return mdp.initial_state()


# --- __init__ ---

def test_agent_initial_epsilon(agent):
    assert agent.epsilon == 1.0


def test_agent_q_table_starts_empty(agent):
    assert len(agent.q_table) == 0


# --- _q ---

def test_q_returns_zero_for_unseen_state(agent, initial):
    assert agent._q(initial, 0) == 0.0


def test_q_returns_stored_value(agent, initial):
    agent.q_table[(initial, 3)] = 0.7
    assert agent._q(initial, 3) == 0.7


# --- best_action ---

def test_best_action_returns_highest_q(agent, initial):
    agent.q_table[(initial, 0)] = 0.1
    agent.q_table[(initial, 3)] = 0.9
    agent.q_table[(initial, 6)] = 0.4
    assert agent.best_action(initial, [0, 3, 6]) == 3


def test_best_action_only_considers_legal(agent, initial):
    # col 3 has highest Q but is not legal
    agent.q_table[(initial, 3)] = 0.9
    agent.q_table[(initial, 1)] = 0.5
    assert agent.best_action(initial, [0, 1, 2]) == 1


# --- select_action ---

def test_select_action_always_explores_at_epsilon_1(agent, initial):
    random.seed(42)
    legal = [0, 1, 2, 3, 4, 5, 6]
    # With epsilon=1.0, every call is random — just check it returns a legal action
    for _ in range(20):
        action = agent.select_action(initial, legal)
        assert action in legal


def test_select_action_always_exploits_at_epsilon_0(agent, initial):
    agent.epsilon = 0.0
    agent.q_table[(initial, 4)] = 1.0
    for _ in range(10):
        assert agent.select_action(initial, [0, 1, 4, 6]) == 4


# --- update ---

def test_update_creates_q_entry(agent, mdp, initial):
    next_state = mdp.step(initial, 3)
    agent.update(initial, 3, 0.0, next_state, mdp.get_actions(next_state), done=False)
    assert (initial, 3) in agent.q_table


def test_update_terminal_uses_reward_only(agent, mdp):
    # Manufacture a near-win state for PLAYER1 and complete it
    state = mdp.initial_state()
    for col in [0, 1, 0, 1, 0, 1]:   # P1: col0 x3, P2: col1 x3
        state = mdp.step(state, col)
    # One more drop in col 0 should win for PLAYER1
    winning_state = mdp.step(state, 0)
    assert mdp.is_terminal(winning_state)
    reward = mdp.get_reward(winning_state, PLAYER1)
    agent.update(state, 0, reward, winning_state, [], done=True)
    # Q(s, 0) should have moved toward +1.0 from 0.0
    assert agent.q_table[(state, 0)] > 0.0


def test_update_non_terminal_bootstraps_from_next(agent, mdp, initial):
    next_state = mdp.step(initial, 3)
    # Pre-load a positive Q-value in the next state
    agent.q_table[(next_state, 3)] = 1.0
    agent.update(initial, 3, 0.0, next_state, [3], done=False)
    # Q(initial, 3) should be > 0 because it bootstrapped from next
    assert agent.q_table[(initial, 3)] > 0.0


# --- train ---

def test_train_returns_one_reward_per_episode(mdp):
    agent = TabularQAgent(player=PLAYER1)
    rewards = train(agent, mdp, n_episodes=100)
    assert len(rewards) == 100


def test_train_rewards_are_valid_values(mdp):
    agent = TabularQAgent(player=PLAYER1)
    rewards = train(agent, mdp, n_episodes=200)
    assert all(r in {1.0, -1.0, 0.0} for r in rewards)


def test_train_epsilon_decays(mdp):
    agent = TabularQAgent(player=PLAYER1, epsilon=1.0)
    train(agent, mdp, n_episodes=100, epsilon_decay=0.9, epsilon_min=0.05)
    assert agent.epsilon < 1.0


def test_train_epsilon_does_not_go_below_min(mdp):
    agent = TabularQAgent(player=PLAYER1, epsilon=1.0)
    train(agent, mdp, n_episodes=5000, epsilon_decay=0.5, epsilon_min=0.05)
    assert agent.epsilon == pytest.approx(0.05)


def test_train_q_table_grows(mdp):
    agent = TabularQAgent(player=PLAYER1)
    train(agent, mdp, n_episodes=200)
    assert len(agent.q_table) > 0
