# src/utils/environment.py
import tmrl

def make_env():
    env = tmrl.get_environment()
    return env