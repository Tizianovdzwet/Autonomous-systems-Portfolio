"""
train_warlords_ram.py — PPO training for Warlords in RAM mode
Matches the tournament notebook exactly: obs_type="ram", 128-byte observations.
Uses MlpPolicy (not CnnPolicy) since input is a flat vector, not pixels.
"""

import numpy as np
import supersuit as ss
from pettingzoo.atari import warlords_v3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
import os


def make_env(n_envs: int = 16):
    # Must match tournament: obs_type="ram"
    env = warlords_v3.parallel_env(obs_type="ram")
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, n_envs, num_cpus=4, base_class="stable_baselines3")
    env = VecMonitor(env)
    # Normalize both obs and reward — important for RAM bytes (0-255 range)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)
    return env


def train(total_timesteps: int = 5_000_000, n_envs: int = 16):
    os.makedirs("checkpoints", exist_ok=True)

    env = make_env(n_envs=n_envs)

    model = PPO(
        "MlpPolicy",          # MLP, not CNN — input is 128-byte flat vector
        env,
        n_steps=512,
        batch_size=512,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,        # entropy bonus to prevent action collapse
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[256, 256]),  # two hidden layers
        verbose=1,
        tensorboard_log="./tb_logs/",
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=250_000,
        save_path="checkpoints/",
        name_prefix="ppo_warlords_ram",
    )

    print(f"Training for {total_timesteps:,} timesteps on RAM observations …")
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb])

    model.save("ppo_warlords_final")
    env.save("ppo_warlords_vecnorm.pkl")
    print("Saved: ppo_warlords_final.zip + ppo_warlords_vecnorm.pkl")

    env.close()
    return model


if __name__ == "__main__":
    train()
