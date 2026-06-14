import numpy as np
from stable_baselines3 import DQN
import os

class Agent1:
    def __init__(self):
        path = os.path.join(os.path.dirname(__file__), "dqn_warlords_ram_2000000_steps")
        self.model = DQN.load(path, device="cpu")

    def act(self, observation) -> int:
        obs = np.asarray(observation, dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)