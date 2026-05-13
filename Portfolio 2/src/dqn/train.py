import os
import random
import wandb
from src.utils.environment import make_env
from src.utils.environment import process_observation
from src.dqn.agent import DQNAgent

# Hyperparameters
LR = 0.001
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05
BUFFER_SIZE = 100000
BATCH_SIZE = 32
HIDDEN_SIZE = 64
EPISODES = 500
TARGET_UPDATE_EVERY = 10
SAVE_EVERY = 50
TRAIN_EVERY = 8


RUN_NAME = f"DQN-lr{LR}-decay{EPSILON_DECAY}-hidden{HIDDEN_SIZE}-batch{BATCH_SIZE}"

run = wandb.init(
    entity="hhs-autonomous-systems",
    project="trackmania-rl",
    name=RUN_NAME,
    config={
        "episodes": EPISODES,
        "lr": LR,
        "gamma": GAMMA,
        "epsilon_decay": EPSILON_DECAY,
        "epsilon_min": EPSILON_MIN,
        "hidden_size": HIDDEN_SIZE,
        "batch_size": BATCH_SIZE,
        "buffer_size": BUFFER_SIZE,
        "target_update_every": TARGET_UPDATE_EVERY,
    }
)

agent = DQNAgent(
    input_size=83,
    hidden_size=HIDDEN_SIZE,
    output_size=4,
    lr=LR,
    gamma=GAMMA,
    epsilon=EPSILON,
    epsilon_decay=EPSILON_DECAY,
    epsilon_min=EPSILON_MIN,
    buffer_size=BUFFER_SIZE,
    batch_size=BATCH_SIZE
)

# Uncomment to resume from checkpoint:
# agent.load(f"experiments/runs/{RUN_NAME}/dqn_episode_X.pt")

env = make_env()
os.makedirs(f"experiments/runs/{RUN_NAME}", exist_ok=True)

ACTIONS = [
    [1, 0, 0],   # forward
    [1, 0, 1],   # forward + left
    [1, 0, -1],  # forward + right
    [0, 1, 0],   # brake
]

for x in range(EPISODES):
    obs, info = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        loss = None
        state = process_observation(obs)
        action = agent.select_action(state)

        next_obs, reward, terminated, truncated, info = env.step(ACTIONS[action])
        next_state = process_observation(next_obs)
        done = terminated or truncated
        total_reward += reward
        step_count += 1

        agent.store_transition(state, action, reward, next_state, done)
        if step_count % TRAIN_EVERY == 0:
            loss = agent.train()
        
        if loss is not None:
            wandb.log({"loss": loss})
        else:
            wandb.log({"loss": 0})

        obs = next_obs

    agent.decay_epsilon()

    if x % TARGET_UPDATE_EVERY == 0:
        agent.update_target_network()

    if x % SAVE_EVERY == 0:
        agent.save(f"experiments/runs/{RUN_NAME}/dqn_episode_{x}.pt")

    wandb.log({
        "episode": x,
        "total_reward": total_reward,
        "epsilon": agent.epsilon,
        "buffer_size": len(agent.buffer),
        "episode_length": step_count,
    })
    print(f"Episode {x} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f} | Steps: {step_count}")

agent.save(f"experiments/runs/{RUN_NAME}/dqn_final.pt")
wandb.finish()