# Portfolio 3 — Multi-Agent Reinforcement Learning: Warlords

This project implements a **Deep Q-Network (DQN)** agent for the Atari multi-agent game [Warlords](https://ale.farama.org/multi-agent-environments/warlords/) using RAM-based observations. The agent is trained via independent learners using Stable-Baselines3 and PettingZoo, and competes against other agents in a class tournament.

---

## Project Structure

```
portfolio 3/
├── agent1.py                        # Trained DQN agent (tournament entry)
├── agent2.py                        # Random baseline agent (placeholder)
├── agent3.py                        # Random baseline agent (placeholder)
├── agent4.py                        # Random baseline agent (placeholder)
├── Train_warlords.ipynb             # DQN training notebook (runs on Google Colab)
├── dqn_warlords_ram_1500000_steps.zip  # Trained model weights (generated after training)
├── warlords_tournament_ram_mode.ipynb  # Tournament notebook (provided)
├── Requirements.txt                 # Local dependencies
└── README.md                        # This file
```

---

## Environment

**Warlords (v3)**: 4-player competitive game; 128-byte RAM observations, 6 discrete actions per agent.

---

## Algorithm

**Deep Q-Network (DQN)** with MLP policy:
- Experience replay improves data efficiency in sparse-reward environment
- Discrete action space (6 actions) matches DQN's design
- Flat RAM observations (128 bytes) require MLP, not CNN
- Independent learners: single paddle trained against 3 random agents

### Final Hyperparameters (Config C — 1.5M steps)

| Parameter | Value | Motivation |
|---|---|---|
| `learning_rate` | 1e-4 | Lower lr gives more stable Q-value updates |
| `buffer_size` | 100,000 | Larger buffer reduces sample correlation |
| `batch_size` | 32 | Standard mini-batch size for DQN |
| `gamma` | 0.99 | High discount — long-term strategy matters |
| `train_freq` | 4 | Update every 4 environment steps |
| `target_update_interval` | 1,000 | Stable target network updates |
| `exploration_fraction` | 0.10 | Epsilon decays over first 10% of training |
| `exploration_final_eps` | 0.02 | Minimum exploration rate |
| `total_timesteps` | 1,500,000 | Training budget (convergence achieved) |

---

## Setup

### Training (Google Colab)

1. Upload `Train_warlords.ipynb` to [Google Colab](https://colab.research.google.com)
2. Set runtime to GPU (T4 recommended)
3. Run all cells (~30–60 minutes)
4. Download `dqn_warlords_ram_1500000_steps.zip` to `portfolio 3/`

### Running Tournament Locally

```bash
pip install -r Requirements.txt
python -c "from warlords_tournament_ram_mode import run_tournament; run_tournament()"
```

---

## Running Tournament

```
portfolio 3/
├── warlords_tournament_ram_mode.ipynb
├── agent1.py .. agent4.py
└── dqn_warlords_ram_1500000_steps.zip
```

Open the notebook and run all cells.

---

## Results

Training reward curves, hyperparameter experiments, and baseline comparison are documented in `Train_warlords.ipynb`.

**Key findings:**
- Three hyperparameter configurations (A, B, C) were tested iteratively
- Config C (lr=1e-4, buffer=100k, 1.5M steps) achieved convergence
- Sanity check confirms the policy produces all 6 discrete actions (no mode collapse)
- **Baseline comparison:** DQN achieves 25% win rate vs. random agent's 15% in the same position (expected: 25% by chance)
- **Conclusion:** DQN successfully learns a policy better than random, demonstrating that RL capture exploit structure in the Warlords environment despite sparse rewards

---

## References

> *Full APA reference list to be added in the report.*

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347
- Terry, J., Black, B., et al. (2021). *PettingZoo: Gym for Multi-Agent Reinforcement Learning.* NeurIPS.
- Raffin, A., Hill, A., et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations.* JMLR.