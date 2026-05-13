import random
import wandb
from src.utils.environment import make_env

episodes = 10

run = wandb.init(
    entity="hhs-autonomous-systems",
    project="trackmania-rl",
    name="random-policy",
    config={
        "policy": "random",
        "episodes": episodes,
    }
)

env = make_env()

ACTIONS = [
    [1, 0, 0],  # forward
    [1, 0, 1],  # forward + left
    [1, 0, -1], # forward + right
    [0, 1, 0],  # brake
]

for x in range(episodes):
    env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = random.choice(ACTIONS)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
    
    wandb.log({"episode": x, "total_reward": total_reward})
    print(f"Episode {x}, Total Reward: {total_reward}")

wandb.finish()