import numpy as np
from stable_baselines3 import DQN

class Agent1:
    def __init__(self):
        self.model = DQN.load("dqn_warlords_ram_2000000_steps", device="cpu")

    def act(self, observation) -> int:
        obs = np.asarray(observation, dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)