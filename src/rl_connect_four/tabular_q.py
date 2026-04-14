"""
Tabular Q-Learning for Connect Four.

The Q-table is a dictionary: (State, action) -> float.
Because State is a frozen dataclass (hashable), it works as a dick key directly.

We train PLAYER1 against a random opponent (PLAYER2). THis is the simplest possible
setup and will hit the state-space wall quickly as we have about 4.5 trillion
states and each have 7 actions so 31.5 trillion entries.


Note:
    - The Q-table is the knowledge — it stores estimated values for every (state, action) pair the   
    agent has seen
    - The policy is the decision rule — given a state, which action do I pick?                       
                                                                                                
    In Q-learning, the policy is derived from the Q-table: "pick the action with the highest         
    Q-value." That's it. The Q-table fully defines the greedy policy.                                
                                                                                                
    Epsilon-greedy is just a wrapper around that greedy policy that says "but sometimes, ignore the  
    table and pick randomly instead.

"""

import random
from collections import defaultdict

from rl_connect_four.mdp import ConnectFourMDP, State
from rl_connect_four.engine import PLAYER1, PLAYER2, other_player


class TabularQAgent:
    """
    Q-Learning agent backed by an explicit lookup table.
                                                                                                                                                                        
    Hyperparameters:
    alpha    — learning rate: how much each update shifts the Q-value                                                                                               
    gamma    — discount factor: how much future rewards are worth today                                                                                             
    epsilon  — exploration rate: probability of picking a random action
    """

    def __init__(self, player: int, alpha: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 1.0):
        self.player = player
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table: (State, action) -> Q-value
        # We use defaultdict to return 0.0 for unseen (State, action) pairs.
        self.q_table: dict[tuple[State, int], float] = defaultdict(float)


    def _q(self, state: State, action: int) -> float:
        """Return the Q-value for (state, action), defaulting to 0.0 if unseen."""
        return self.q_table[(state, action)]
    

    def best_action(self, state: State, legal_actions:list[int]) -> int:
        """Return the action with the highest Q-value (greedy policy)"""
        return max(legal_actions, key= lambda a : self._q(state, a))
    

    def select_action(self, state: State, legal_actions: list[int]) -> int:
        """
        Epsilon-greedy policy.
        With probability epsilon: pick a random legal action (explore).
        Otherwise: pick the action with the highest Q-value (exploit)

        Exploration is critical early in the training because the Q-table is all
        zeros, so greedy would just pick the first column forever.
        """

        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        return self.best_action(state, legal_actions)
    

    def update(self, state: State, action: int, reward: float,
               next_state: State, next_legal_actions: list[int],
               done: bool) -> None:
        """
        Update Q(state, action) using the Bellman Equation
            Q(s, a) <- Q(s, a) + alpha * [target - Q(s, a)]
                                                                                                   
        where target = r                                if terminal
                     = r + gamma * max_a' Q(s', a')     otherwise
        """
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(
                self._q(next_state, a) 
                for a in next_legal_actions
                )
        
        # TD -> Temporal Difference
        td_error = target - self._q(state, action)
        self.q_table[(state, action)] += self.alpha * td_error


def train(
        agent: TabularQAgent,
        mdp: ConnectFourMDP,
        n_episodes: int = 50_000,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.05
        ) -> list[float]:
    """
    Train the agent by playing n_episodes games against a random opponent.
    PLAYER2 always picks randomly - the agent learns to exploit that.

    Returns a list of rewards (for the agent's perpective) for each episode
    usedul for plotting learning progress.
    """
    
    rewards = []

    for episode in range(n_episodes):
        state = mdp.initial_state()

        while not mdp.is_terminal(state):
            legal_actions = mdp.get_actions(state)

            if state.current_player == agent.player:
                action = agent.select_action(state, legal_actions)
            else:
                action = random.choice(legal_actions) # random opponent
            
            next_state = mdp.step(state, action)
            done = mdp.is_terminal(next_state)

            if state.current_player == agent.player:
                reward = mdp.get_reward(next_state, agent.player) if done else 0.0
                next_legal = mdp.get_actions(next_state) if not done else []
                agent.update(state, action, reward, next_state, next_legal, done)
            
            state = next_state
        
        rewards.append(reward)

        # Decay episilon afer each episode
        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)
        
    return rewards





    


        


