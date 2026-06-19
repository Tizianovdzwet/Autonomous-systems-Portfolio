import os
import numpy as np
from stable_baselines3 import DQN


class Agent1:
    """Trained DQN agent for Warlords."""

    def __init__(self):
        path = os.path.join(os.path.dirname(__file__), "dqn_warlords_ram_1500000_steps")
        self.model = DQN.load(path, device="cpu")

    def act(self, observation) -> int:
        """Return action based on observation using trained DQN policy."""
        obs = np.asarray(observation, dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)