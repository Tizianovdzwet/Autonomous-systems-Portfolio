"""
agent1.py — RAM-mode Warlords agent
obs is a 128-byte numpy array (uint8), matching obs_type="ram" in the tournament.
"""

import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from gymnasium import spaces


class Agent1:
    def __init__(self):
        self.model = PPO.load("ppo_warlords_final", device="cpu")
        self._vecnorm = None

        if os.path.exists("ppo_warlords_vecnorm.pkl"):
            dummy = DummyVecEnv([lambda: _DummyRAMEnv()])
            self._vecnorm = VecNormalize.load("ppo_warlords_vecnorm.pkl", dummy)
            self._vecnorm.training = False
            self._vecnorm.norm_reward = False

    def act(self, observation) -> int:
        obs = np.asarray(observation, dtype=np.float32)

        if self._vecnorm is not None:
            obs = self._vecnorm.normalize_obs(obs[None])[0]

        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


class _DummyRAMEnv(gym.Env):
    """Placeholder so VecNormalize can be instantiated with the right obs shape."""
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(0, 255, shape=(128,), dtype=np.uint8)
        self.action_space = spaces.Discrete(18)

    def reset(self, **kwargs):
        return np.zeros(128, dtype=np.uint8), {}

    def step(self, action):
        return np.zeros(128, dtype=np.uint8), 0.0, True, False, {}