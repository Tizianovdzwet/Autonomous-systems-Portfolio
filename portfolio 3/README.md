# Portfolio 3 — Multi-Agent Reinforcement Learning: Warlords

This project implements a **Deep Q-Network (DQN)** agent for the Atari multi-agent game [Warlords](https://ale.farama.org/multi-agent-environments/warlords/) using RAM-based observations. The agent is trained via independent learners using Stable-Baselines3 and PettingZoo, and competes against other agents in a class tournament.

---

## Project Structure

```
portfolio 3/
├── agent1.py                        # Trained PPO agent (tournament entry)
├── agent2.py                        # Random baseline agent (placeholder)
├── agent3.py                        # Random baseline agent (placeholder)
├── agent4.py                        # Random baseline agent (placeholder)
├── train.ipynb                      # PPO training notebook (runs on Google Colab)
├── ppo_warlords_final.zip           # Trained model weights (generated after training)
├── warlords_tournament_ram_mode.ipynb  # Tournament notebook (provided)
├── Requirements.txt                 # Local dependencies
└── README.md                        # This file
```

---

## Environment

The agent plays **Warlords (v3)** via PettingZoo's Atari multi-agent environment:

- **Observation space:** RAM mode — 128 bytes per agent per step
- **Action space:** Discrete (6 actions)
- **Players:** 4 agents competing simultaneously
- **Goal:** Defend your castle wall while destroying opponents'

---

## Algorithm

The agent uses **DQN (Deep Q-Network)** with an MLP policy, chosen for the following reasons:

- RAM observations are flat 128-byte vectors — no spatial structure — making MLP more appropriate than CNN
- DQN's experience replay buffer improves data efficiency, important in the sparse-reward Warlords environment
- The discrete action space (6 actions) is exactly what DQN is designed for
- Stable-Baselines3 provides a well-tested DQN implementation with GPU support

The multi-agent environment is wrapped with SuperSuit so DQN can train on a single paddle's perspective (independent learners). Three hyperparameter configurations were tested before arriving at the final model (see `Train_warlords.ipynb`).

### Final Hyperparameters (Config C — 5M steps)

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
| `total_timesteps` | 5,000,000 | Training budget |

---

## Setup

### Prerequisites

- Python 3.11
- Conda (recommended) or pip
- Windows: enable long paths and use a short temp directory (see below)

### Local installation (for running the tournament notebook)

```bash
conda create -n warlords python=3.11 -y
conda activate warlords
conda install -c conda-forge zlib -y
$env:CMAKE_PREFIX_PATH = "C:\Users\<you>\miniconda3\envs\warlords\Library"  # Windows only
pip install -r Requirements.txt
```

### Training (Google Colab)

Training is done on Google Colab due to Windows compatibility issues with `multi_agent_ale_py`. Open `Train_warlords.ipynb` in Colab and run all cells. The trained model is saved to Google Drive as `dqn_warlords_ram_5000000_steps.zip`.

1. Upload `Train_warlords.ipynb` to [Google Colab](https://colab.research.google.com)
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells — training takes approximately 2–4 hours for 5M steps
4. Download `dqn_warlords_ram_5000000_steps.zip` and place it in the `portfolio 3/` folder

---

## Running the Tournament

Place all agent files and `ppo_warlords_final.zip` in the same folder as `warlords_tournament_ram_mode.ipynb`, then open and run the notebook.

```
portfolio 3/
├── warlords_tournament_ram_mode.ipynb
├── agent1.py
├── agent2.py
├── agent3.py
├── agent4.py
└── dqn_warlords_ram_5000000_steps.zip   ← required for agent1.py to load
```

---

## Results

Training reward curves, hyperparameter experiments, and baseline comparison are documented in `Train_warlords.ipynb`.

Summary:
- Three hyperparameter configurations (A, B, C) were tested — Config C (5M steps, buffer=100k, lr=1e-4) gave the most stable training
- Sanity check confirms the policy produces 3+ unique actions (no collapse)
- Baseline comparison evaluates DQN vs. random policy over 20 games

---

## File Descriptions

| File | Description |
|---|---|
| `agent1.py` | Loads `dqn_warlords_ram_5000000_steps.zip` and exposes an `act(observation)` method |
| `agent2-4.py` | Random policy placeholders for local testing and baseline comparison |
| `Train_warlords.ipynb` | Full DQN training pipeline with hyperparameter experiments and evaluation |
| `Requirements.txt` | Minimal local dependencies for running the tournament |
| `warlords_tournament_ram_mode.ipynb` | Provided tournament runner (not modified) |

---

## References

> *Full APA reference list to be added in the report.*

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347
- Terry, J., Black, B., et al. (2021). *PettingZoo: Gym for Multi-Agent Reinforcement Learning.* NeurIPS.
- Raffin, A., Hill, A., et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations.* JMLR.