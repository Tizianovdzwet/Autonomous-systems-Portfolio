import numpy as np
from stable_baselines3 import PPO


class Agent1:
    def __init__(self, model_path: str = "ppo_warlords_final"):
        self.model = PPO.load(model_path, device="cpu")

    def act(self, observation: np.ndarray) -> int:
        observation = np.asarray(observation, dtype=np.float32)
        action, _ = self.model.predict(observation[np.newaxis, :], deterministic=True)
        return int(action[0])