# Tic-Tac-Toe AI using MDPs & Reinforcement Learning

The project implements model-based planning agents that solve a fully known MDP using value and policy iteration, alongside model-free reinforcement learning agents that learn optimal play through self-play.

---

## AI Techniques Implemented
- Value Iteration (Offline Planning with known MDP)
- Policy Iteration (Policy Evaluation + Improvement)
- Q-Learning (Model-free Reinforcement Learning)
- ε-greedy exploration
- Terminal state handling
- Policy extraction from value functions
---
## Environment Overview
- Game: 3×3 Tic-Tac-Toe
- Agent role: X player (learning agent)
- Opponent (O): Modelled as part of the environment
- State: Board configuration
- Actions: Legal moves
- Rewards: Assigned on entering states
- Terminal states:
  --X win
  --O win
  --Draw
---
## Agent Descriptions
### Value Iteration Agent
- Implements k-step Value Iteration
- Uses a full MDP model of the environment
- Computes optimal state values
- Extracts a greedy policy from the value function
- Designed as an offline planner

### Policy Iteration Agent
- Starts with a random policy
- Alternates between:
 --Policy Evaluation
 --Policy Improvement
- Converges to an optimal policy

### Q-Learning Agent
- Model-free Reinforcement Learning
- Learns directly from gameplay experience
- Uses ε-greedy exploration during training
- Extracts a deterministic policy after training
---
## Evaluation

Each agent was evaluated by playing 50 games against the following rule-based opponents:
- random
- aggressive
- defensive
Metrics recorded:
- Wins
- Losses
- Draws

  
