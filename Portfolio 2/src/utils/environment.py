# src/utils/environment.py
import tmrl
import numpy as np

def make_env():
    env = tmrl.get_environment()
    return env

def process_observation(obs):
    return np.concatenate([np.array(o).flatten() for o in obs])