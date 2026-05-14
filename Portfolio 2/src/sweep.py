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
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 50
    },
    "parameters": {
        "lr": {
            "values": [0.001, 0.0001, 0.00001]
        },
        "epsilon_decay": {
            "values": [0.990, 0.995, 0.999]
        },
        "hidden_size": {
            "values": [32, 64, 128]
        },
        "batch_size": {
            "values": [16, 32, 64]
        },
        "gamma": {
            "values": [0.95, 0.99, 0.999]
        }
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TRAIN_EVERY = 8 if device.type == "cuda" else 16
TRAIN_EVERY = 16  # simplified
print(f"Using device: {device} | TRAIN_EVERY: {TRAIN_EVERY}")

env = make_env()

def train():
    run = wandb.init()
    config = wandb.config

    RUN_NAME = f"sweep-lr{config.lr}-decay{config.epsilon_decay}-hidden{config.hidden_size}-batch{config.batch_size}-gamma{config.gamma}"
    print(f"\nStarting run: {RUN_NAME}")

    wandb.config.update({
        "run_name": RUN_NAME,
        "resume_command": f"python -m src.sweep {wandb.run.sweep_id}"
    }, allow_val_change=True)

    wandb.run.name = f"phase-1-sweep-{wandb.run.id[:4]}"
    wandb.run.save()

    agent = DQNAgent(
        input_size=83,
        hidden_size=config.hidden_size,
        output_size=4,
        lr=config.lr,
        gamma=config.gamma,
        epsilon=1.0,
        epsilon_decay=config.epsilon_decay,
        epsilon_min=0.05,
        buffer_size=100000,
        batch_size=config.batch_size
    )

    # Warm up GPU/CPU before real-time loop
    dummy = torch.zeros(config.batch_size, 83).to(agent.device)
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

    EPISODES = 500
    TARGET_UPDATE_EVERY = 10
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

            if x % 50 == 0:
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
            })

            print(f"Episode {x} | Reward: {total_reward:.2f} | Avg: {sum(recent_rewards)/len(recent_rewards):.2f} | Epsilon: {agent.epsilon:.3f}")

        agent.save(f"experiments/runs/{RUN_NAME}/dqn_final.pt")
        wandb.finish(exit_code=0)  # success

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


# Your PC creates the sweep, buddy's PC joins it
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
    count=10
)