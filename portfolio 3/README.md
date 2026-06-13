# Portfolio 3 — Multi-Agent Reinforcement Learning: Warlords

This project implements a **Proximal Policy Optimization (PPO)** agent for the Atari multi-agent game [Warlords](https://ale.farama.org/multi-agent-environments/warlords/) using RAM-based observations. The agent is trained via self-play using Stable-Baselines3 and PettingZoo, and competes against other agents in a class tournament.

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

The agent uses **PPO (Proximal Policy Optimization)** with an MLP policy, chosen for the following reasons:

- RAM observations are flat vectors — no spatial structure — making MLP more appropriate than CNN
- PPO is more stable than DQN in competitive multi-agent settings due to its clipped objective
- Stable-Baselines3 provides a well-tested PPO implementation with GPU support

The multi-agent environment is converted to a single-agent vectorised environment using SuperSuit wrappers, allowing standard SB3 training via self-play (the agent learns by playing all four paddles simultaneously).

### Key Hyperparameters

| Parameter | Value | Motivation |
|---|---|---|
| `n_steps` | 512 | Steps per rollout before update |
| `batch_size` | 256 | Mini-batch size for gradient updates |
| `n_epochs` | 4 | Update passes per rollout |
| `gamma` | 0.99 | High discount — long-term strategy matters |
| `gae_lambda` | 0.95 | Reduces variance in advantage estimates |
| `clip_range` | 0.2 | Standard PPO clipping |
| `ent_coef` | 0.01 | Entropy bonus to encourage exploration |
| `learning_rate` | 2.5e-4 | Standard for Atari PPO |
| `total_timesteps` | 2,000,000 | Training budget |

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

Training is done on Google Colab due to Windows compatibility issues with `multi_agent_ale_py`. Open `train.ipynb` in Colab and run all cells. The trained model is saved to Google Drive as `ppo_warlords_final.zip`.

1. Upload `train.ipynb` to [Google Colab](https://colab.research.google.com)
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells — training takes approximately 30–60 minutes
4. Download `ppo_warlords_final.zip` and place it in the `portfolio 3/` folder

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
└── ppo_warlords_final.zip   ← required for agent1.py to load
```

---

## Results

> *To be completed after training.*

Training reward curves and win rates against random baseline agents will be added here, including:

- Episode reward over time (TensorBoard)
- Win rate vs. random agents over 10 games
- Discussion of hyperparameter experiments

---

## File Descriptions

| File | Description |
|---|---|
| `agent1.py` | Loads `ppo_warlords_final.zip` and exposes an `act(observation)` method |
| `agent2-4.py` | Random policy placeholders for local testing |
| `train.ipynb` | Full PPO training pipeline with documentation |
| `Requirements.txt` | Minimal local dependencies for running the tournament |
| `warlords_tournament_ram_mode.ipynb` | Provided tournament runner (not modified) |

---

## References

> *Full APA reference list to be added in the report.*

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347
- Terry, J., Black, B., et al. (2021). *PettingZoo: Gym for Multi-Agent Reinforcement Learning.* NeurIPS.
- Raffin, A., Hill, A., et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations.* JMLR.