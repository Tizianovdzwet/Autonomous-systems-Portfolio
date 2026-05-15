import os
import torch
import wandb
from src.utils.environment import make_env
from src.utils.environment import process_observation
from src.dqn.agent import DQNAgent

# ============================================================
# EXPERIMENT PARAMETERS - change these between runs
# ============================================================
LR = 0.00001
EPSILON_DECAY = 0.999
GAMMA = 0.95
TARGET_UPDATE_EVERY = 50

# ============================================================
# FIXED PARAMETERS - don't change these between runs
# ============================================================
EPSILON = 1.0
EPSILON_MIN = 0.05
BUFFER_SIZE = 500000
BATCH_SIZE = 32
HIDDEN_SIZE = 64
EPISODES = 5000
SAVE_EVERY = 200
TRAIN_EVERY = 16
INPUT_SIZE = 83
N_ACTIONS = 4

# ============================================================
# SETUP
# ============================================================
RUN_NAME = f"final-lr{LR}-decay{EPSILON_DECAY}-gamma{GAMMA}-tgt{TARGET_UPDATE_EVERY}-3000ep"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | TRAIN_EVERY: {TRAIN_EVERY}")

run = wandb.init(
    entity="hhs-autonomous-systems",
    project="trackmania-rl",
    name=RUN_NAME,
    config={
        "lr": LR,
        "epsilon_decay": EPSILON_DECAY,
        "gamma": GAMMA,
        "target_update_every": TARGET_UPDATE_EVERY,
        "episodes": EPISODES,
        "epsilon_min": EPSILON_MIN,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "train_every": TRAIN_EVERY
    }
)

agent = DQNAgent(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    output_size=N_ACTIONS,
    lr=LR,
    gamma=GAMMA,
    epsilon=EPSILON,
    epsilon_decay=EPSILON_DECAY,
    epsilon_min=EPSILON_MIN,
    buffer_size=BUFFER_SIZE,
    batch_size=BATCH_SIZE
)

# Warm up GPU before real-time loop
dummy = torch.zeros(BATCH_SIZE, INPUT_SIZE).to(agent.device)
with torch.no_grad():
    agent.q_network(dummy)
    agent.target_network(dummy)
print("Network warmed up!")

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

# ============================================================
# TRAINING LOOP
# ============================================================
recent_rewards = []
x = 0

try:
    for x in range(EPISODES):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        loss = None

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

            obs = next_obs

        agent.decay_epsilon()

        if x % TARGET_UPDATE_EVERY == 0:
            agent.update_target_network()

        if x % SAVE_EVERY == 0:
            agent.save(f"experiments/runs/{RUN_NAME}/dqn_episode_{x}.pt")

        recent_rewards.append(total_reward)
        if len(recent_rewards) > 50:
            recent_rewards.pop(0)

        wandb.log({
            "episode": x,
            "total_reward": total_reward,
            "avg_recent_reward": sum(recent_rewards) / len(recent_rewards),
            "epsilon": agent.epsilon,
            "buffer_size": len(agent.buffer),
            "episode_length": step_count,
            "loss": loss if loss is not None else 0,
        }, step=x)

        print(f"Episode {x} | Reward: {total_reward:.2f} | Avg: {sum(recent_rewards)/len(recent_rewards):.2f} | Epsilon: {agent.epsilon:.3f} | Steps: {step_count}")

    agent.save(f"experiments/runs/{RUN_NAME}/dqn_final.pt")
    wandb.finish(exit_code=0)

except KeyboardInterrupt:
    print("Manual stop - saving checkpoint...")
    agent.save(f"experiments/runs/{RUN_NAME}/dqn_episode_{x}_stopped.pt")
    wandb.finish(exit_code=1)
    raise
except BaseException as e:
    print(f"Run crashed at episode {x}: {e}")
    agent.save(f"experiments/runs/{RUN_NAME}/dqn_episode_{x}_crashed.pt")
    wandb.finish(exit_code=1)
finally:
    pass