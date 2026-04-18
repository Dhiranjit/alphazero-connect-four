"""
Deep Q-Network (DQN) for Connect Four.
Two problems with tabular Q-learning motivated this:                                           
    1. Memory: 4.5 trillion states x 7 actions can't fit in a dictionary.                          
    2. Generalisation: every new board position is treated as completely unknown,                  
       even if it differs from a seen position by one piece.  

A neural network fixes both: it maps any board state → Q-values using learned                    
weights, generalising across similar positions and fitting in constant memory.                   
                                                                                                
Two instabilities arise from simply replacing the table with a network:                          
- Correlated samples: consecutive game transitions are highly correlated,                      
    which causes gradient updates to overfit recent experience.                                
- Moving targets: the target value r + γ·max Q(s') also changes every step                     
    because Q is the same network we're updating.                                             
                                                                                                
Both are fixed by:                                                                               
- Experience replay: store transitions in a buffer, sample random minibatches.                 
- Target network: a separate, periodically-synced copy of Q used only for                    
    computing bootstrap targets. (Updated every N steps)
"""

import random
import collections

import torch
import torch.nn as nn

from rl_connect_four.engine import PLAYER1, PLAYER2
from rl_connect_four.mdp import State


BOARD_ROWS = 6
BOARD_COLS = 7
INPUT_SIZE = BOARD_ROWS * BOARD_COLS  # 42 cells
HIDEN_SIZE = 128

Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'valid_next_actions', 'done'))

def encode_state(state: State) -> torch.Tensor:
    """
    Encode a board state as a 42-float vector from the current player's perspective.

    My pieces: 1.0, opponent's pieces: -1.0, empty: 0.0

    This symmetry allows the model to see myside vs their side, regardless of
    which player is acting.
    """

    current = state.current_player
    opponent = PLAYER2 if current == PLAYER1 else PLAYER1

    cells = []
    for row in state.board:
        for cell in row:
            if cell == current:
                cells.append(1.0)
            elif cell == opponent:
                cells.append(-1.0)
            else:
                cells.append(0.0)

    return torch.tensor(cells, dtype=torch.float32)


class QNetwork(nn.Module):
    """
    Maps a board state (vector of length 42) to Q-values for each of the 7 columns.
    Two hidden layers of 128 units - enough capacity to learn positional patterns
    without being so large that it overfits the replay buffer.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDEN_SIZE, HIDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDEN_SIZE, BOARD_COLS)
        )
    
    def forward(self, x):
        return self.net(x)
    

class ReplayBuffer:
    """
    Circulat buffer of past transitions. Sampling random minibatches from here 
    breaks the temporal correlation between consecutive game step - without this,
    gradient updates would overfit to whatever happened in the game.
    """
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, valid_next_actions, done):
        self.buffer.append(Transition(state, action, reward, next_state, valid_next_actions, done))
    
    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    """
    DQN agent for Connect Four.

    Maintains two networks:
    - online: trained every step via gradient descent
    - target: frozen copy, periodically synced from online, used only for 
    computing Bellman targets - prevents the moving-target instability
    """

    def __init__(
            self,
            lr: float = 1e-3,
            gamma: float = 0.99,
            epsilon: float = 1.0,
            epsilon_min: float = 0.05,
            epsilon_decay: float = 0.9985,
            buffer_capacity: int = 20_000,
            batch_size: int = 64,
            target_update_freq: int = 500
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps = 0  # counts gradient updates, triggers target sync

        self.online = QNetwork()
        self.target = QNetwork()
        self.target.load_state_dict(self.online.state_dict()) # start in sync
        self.target.eval() # since target is never trained directly

        self.buffer = ReplayBuffer(buffer_capacity)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)


    def select_action(self, state: torch.Tensor, valid_actions: list[int]) -> int:
        """
        Epsilon-greedy action selection over valid columns.
        With probability epsilonL explore (random valid columns).
        Otherwise: exploit (column with highest Q-value among valid ones).
        """
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        with torch.inference_mode():
            q_values = self.online(state.unsqueeze(0)).squeeze(0) # shape: (7,)
        
        # Mask invalid actions with -inf so they've never picked as argmax
        mask = torch.full((BOARD_COLS,), float('-inf'))
        mask[list(valid_actions)] = 0.0
        return (q_values + mask).argmax().item()
    

    def store(self, state, action, reward, next_state, valid_next_actions, done):
        self.buffer.push(state, action, reward, next_state, valid_next_actions, done)


    def update(self):
        """
        Sample a random minibatch and do one gradient update on the online network.
        """
        if len(self.buffer) < self.batch_size:
            return # not enough experience yet
        
        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        states      = torch.stack(batch.state)                          # (B, 42)
        actions     = torch.tensor(batch.action, dtype=torch.long)      # (B,)
        rewards     = torch.tensor(batch.reward, dtype=torch.float32)   # (B,)
        next_states = torch.stack(batch.next_state)                     # (B, 42)
        dones       = torch.tensor(batch.done, dtype=torch.float32)     # (B,)

        # Q(s, a) from the online network - for the column that we actually played
        q_pred = self.online(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze(1) # (B,)

        # Double DQN target:
        #   - online net selects the best next action (which looks best?)
        #   - target net evaluates that action (what's it really worth?)
        # Decoupling selection from evaluation breaks the max-operator bias
        # that causes vanilla DQN to systematically overestimate Q-values.
        with torch.inference_mode():
            q_next_online = self.online(next_states)                          # (B, 7)
            mask = torch.full_like(q_next_online, float('-inf'))
            for i, valid in enumerate(batch.valid_next_actions):
                mask[i, list(valid)] = 0.0
            next_actions = (q_next_online + mask).argmax(dim=1, keepdim=True)  # (B, 1)

            q_next_target = self.target(next_states)                           # (B, 7)
            q_next = q_next_target.gather(1, next_actions).squeeze(1)          # (B,)

        q_next = torch.where(dones.bool(), torch.zeros_like(q_next), q_next)   # terminal: no bootstrap
        targets = rewards + self.gamma * q_next

        loss = nn.functional.mse_loss(q_pred, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Sync target network every target_update_freq steps
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

    def decay_epsilon(self):
        """Decay epsilon once per episode, after all gradient updates for that episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            

            