import os
import sys
import torch
import wandb
from src.utils.environment import make_env
from src.utils.environment import process_observation
from src.dqn.agent import DQNAgent

sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "avg_recent_reward",
        "goal": "maximize"
    },
    "parameters": {
        "lr": {
            "values": [0.000005, 0.00001, 0.00005]
        },
        "epsilon_decay": {
            "values": [0.997, 0.998, 0.999]
        },
        "gamma": {
            "values": [0.95, 0.99]
        },
        "target_update_every": {
            "values": [10, 20, 50]
        }
    }
}

# Fixed parameters
HIDDEN_SIZE = 64
BATCH_SIZE = 32
EPISODES = 1000
TRAIN_EVERY = 16
SAVE_EVERY = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | TRAIN_EVERY: {TRAIN_EVERY}")

env = make_env()

def train():
    run = wandb.init()
    config = wandb.config

    RUN_NAME = f"p2-lr{config.lr}-decay{config.epsilon_decay}-gamma{config.gamma}-tgt{config.target_update_every}"
    print(f"\nStarting run: {RUN_NAME}")

    wandb.config.update({
        "run_name": RUN_NAME,
        "hidden_size": HIDDEN_SIZE,
        "batch_size": BATCH_SIZE,
        "episodes": EPISODES,
        "train_every": TRAIN_EVERY,
        "resume_command": f"python -m src.sweep2 {wandb.run.sweep_id}"
    }, allow_val_change=True)

    wandb.run.name = f"phase-2-sweep-{wandb.run.id[:4]}"

    agent = DQNAgent(
        input_size=83,
        hidden_size=HIDDEN_SIZE,
        output_size=4,
        lr=config.lr,
        gamma=config.gamma,
        epsilon=1.0,
        epsilon_decay=config.epsilon_decay,
        epsilon_min=0.05,
        buffer_size=100000,
        batch_size=BATCH_SIZE
    )

    dummy = torch.zeros(BATCH_SIZE, 83).to(agent.device)
    with torch.no_grad():
        agent.q_network(dummy)
        agent.target_network(dummy)
    print("Network warmed up!")

    os.makedirs(f"experiments/runs/{RUN_NAME}", exist_ok=True)

    ACTIONS = [
        [1, 0, 0],   # forward
        [1, 0, 1],   # forward + left
        [1, 0, -1],  # forward + right
        [0, 1, 0],   # brake
    ]

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

            if x % config.target_update_every == 0:
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

            print(f"Episode {x} | Reward: {total_reward:.2f} | Avg: {sum(recent_rewards)/len(recent_rewards):.2f} | Epsilon: {agent.epsilon:.3f}")

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


if len(sys.argv) > 1:
    sweep_id = sys.argv[1]
    print(f"Joining existing sweep: {sweep_id}")
else:
    sweep_id = wandb.sweep(
        sweep_config,
        entity="hhs-autonomous-systems",
        project="trackmania-rl"
    )
    print(f"Created sweep: {sweep_id}")

wandb.agent(
    sweep_id,
    function=train,
    entity="hhs-autonomous-systems",
    project="trackmania-rl",
    count=5
)