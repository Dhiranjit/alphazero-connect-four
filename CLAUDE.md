# AlphaZero Connect Four

## What This Project Is

A concept-first reconstruction of reinforcement learning, built around Connect Four. The goal is not to jump to AlphaZero — it is to *rediscover why AlphaZero exists* by hitting the real limitations of each simpler approach in sequence.

Every stage introduces a new idea only because the previous approach concretely failed. That failure must be understood and demonstrated before moving on.

The long-term motivation is to build foundations for RL in less structured domains — particularly reinforcement learning for large language models, where rewards are sparse and the environment is not well-defined.

---

## The Journey & Why Each Stage Exists

### Stage 1: Environment
Model Connect Four as an MDP (states, actions, rewards, transitions, terminals). Build the game engine. This is the foundation every other stage depends on.

### Stage 2: Tabular Q-Learning
The simplest learning approach: a lookup table of (state, action) → value, updated through experience. Builds intuition for the Bellman equation and temporal difference learning. Fails because Connect Four has ~4 trillion possible states — the table cannot fit in memory and will never converge.

### Stage 3: Deep Q-Network (DQN)
Replace the table with a neural network so the agent generalizes across similar states. Introduces instability (moving targets, correlated samples) — fixed with experience replay and a target network. Fails in adversarial settings because Q-learning assumes a stationary environment, but the opponent is also learning.

### Stage 4: Policy Gradient (REINFORCE / Actor-Critic)
Directly optimize the policy (a probability distribution over actions) instead of estimating values. More natural for adversarial games. Fails due to high variance and sample inefficiency — the agent needs many games to get a useful gradient signal.

### Stage 5: Self-Play
Remove the fixed opponent. The agent plays against itself, creating an automatic curriculum: it always faces an opponent exactly at its current skill level. Snapshots of the old network are kept to prevent catastrophic forgetting. This is how AlphaGo Zero and AlphaZero generate training data.

### Stage 6: Monte Carlo Tree Search (MCTS)
Add deliberate planning at decision time. MCTS simulates future game trajectories and picks moves using UCT (Upper Confidence bound for Trees). This is a shift from *reactive* to *deliberate* behavior. Vanilla MCTS uses random rollouts, which is slow and imprecise for complex positions.

### Stage 7: AlphaZero
Combine everything. A single neural network predicts (policy, value) for any position. MCTS uses the policy as a prior to guide search and the value to replace rollouts. Self-play with MCTS generates training data. The network improves → MCTS improves → better self-play data → the network improves further. A closed feedback loop between learning and planning.

---

## Current Stage
**Stage 1: Environment** — branch `stage1-environment`

| # | Stage | Branch | Status |
|---|-------|--------|--------|
| 1 | Environment & MDP formulation | `stage1-environment` | In progress |
| 2 | Tabular Q-Learning | `stage2-tabular-ql` | — |
| 3 | Deep Q-Network | `stage3-dqn` | — |
| 4 | Policy Gradient | `stage4-policy-gradient` | — |
| 5 | Self-Play | `stage5-self-play` | — |
| 6 | Monte Carlo Tree Search | `stage6-mcts` | — |
| 7 | AlphaZero | `stage7-alphazero` | — |

---

## Conventions
- One stage at a time — complete the exit criterion before moving on
- Each stage lives in its own directory: `stage1_environment/`, `stage2_tabular_ql/`, etc.
- Merge the stage branch to `main` when done; cut the next branch from `main`
- Code should be readable and educational — clarity over performance
- When a stage is done, update the status in the table above and update "Current Stage"

## Git Workflow
```
# Start a new stage
git checkout main && git checkout -b stage<N>-<name>

# When stage is complete — merge and move on
git checkout main && git merge stage<N>-<name>
```
