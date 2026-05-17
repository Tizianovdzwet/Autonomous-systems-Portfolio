import os
import sys
import torch
import wandb
from src.utils.environment import make_env
from src.utils.environment import process_observation
from src.dqn.agent import DQNAgent

ACTION_SPACES = {
    "3_basic": {
        "actions": [
            [1, 0, 0],    # forward
            [1, 0, 1],    # hard left
            [1, 0, -1],   # hard right
        ],
        "n_actions": 3
    },
    "4_brake": {
        "actions": [
            [1, 0, 0],    # forward
            [1, 0, 1],    # hard left
            [1, 0, -1],   # hard right
            [0, 1, 0],    # brake
        ],
        "n_actions": 4
    },
    "5_slight_steer": {
        "actions": [
            [1, 0, 0],    # forward
            [1, 0, 0.5],  # slight left
            [1, 0, -0.5], # slight right
            [1, 0, 1],    # hard left
            [1, 0, -1],   # hard right
        ],
        "n_actions": 5
    },
    "6_full": {
        "actions": [
            [1, 0, 0],    # forward
            [1, 0, 1],    # hard left
            [1, 0, -1],   # hard right
            [0, 1, 0],    # brake
            [0, 1, 1],    # brake + hard left
            [0, 1, -1],   # brake + hard right
        ],
        "n_actions": 6
    },
    "7_full_slight": {
        "actions": [
            [1, 0, 0],    # forward
            [1, 0, 0.5],  # slight left
            [1, 0, -0.5], # slight right
            [1, 0, 1],    # hard left
            [1, 0, -1],   # hard right
            [0, 1, 1],    # brake + hard left
            [0, 1, -1],   # brake + hard right
        ],
        "n_actions": 7
    },
    "8_full_slight_brake": {
        "actions": [
            [1, 0, 0],    # forward
            [1, 0, 0.5],  # slight left
            [1, 0, -0.5], # slight right
            [1, 0, 1],    # hard left
            [1, 0, -1],   # hard right
            [0, 1, 0],    # brake
            [0, 1, 1],    # brake + hard left
            [0, 1, -1],   # brake + hard right
        ],
        "n_actions": 8
    },
    "4_no_brake_slight": {
        "actions": [
            [1, 0, 0],    # forward
            [1, 0, 0.5],  # slight left
            [1, 0, -0.5], # slight right
            [1, 0, 1],    # hard left — asymmetric, track curves left more
        ],
        "n_actions": 4
    },
    "2_minimal": {
        "actions": [
            [1, 0, 1],    # forward + left
            [1, 0, -1],   # forward + right
        ],
        "n_actions": 2
    },
    "3_coast": {
        "actions": [
            [1, 0, 0],    # forward
            [0, 0, 1],    # coast left (no gas, no brake)
            [0, 0, -1],   # coast right
        ],
        "n_actions": 3
    },
    "5_brake_slight": {
        "actions": [
            [1, 0, 0],    # forward
            [1, 0, 1],    # hard left
            [1, 0, -1],   # hard right
            [0, 1, 0.5],  # brake + slight left
            [0, 1, -0.5], # brake + slight right
        ],
        "n_actions": 5
    }
}

sweep_config = {
    "method": "grid",
    "metric": {
        "name": "avg_recent_reward",
        "goal": "maximize"
    },
    "parameters": {
        "action_space": {
            "values": [
                "3_basic",
                "4_brake", 
                "5_slight_steer",
                "6_full",
                "7_full_slight",
                "8_full_slight_brake",
                "4_no_brake_slight",
                "2_minimal",
                "3_coast",
                "5_brake_slight"
            ]
        }
    }
}

# Fixed parameters
HIDDEN_SIZE = 64
BATCH_SIZE = 32
EPISODES = 2000
TRAIN_EVERY = 16
SAVE_EVERY = 200
EPSILON = 1.0
EPSILON_MIN = 0.15
BUFFER_SIZE = 500000
GAMMA = 0.99
TARGET_UPDATE_EVERY = 10
LR = 0.000005
EPSILON_DECAY = 0.995

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | TRAIN_EVERY: {TRAIN_EVERY}")

env = make_env()

def train():
    run = wandb.init()
    config = wandb.config

    space = ACTION_SPACES[config.action_space]
    ACTIONS = space["actions"]
    N_ACTIONS = space["n_actions"]

    RUN_NAME = f"p3-{[config.action_space]}"
    print(f"\nStarting run: {RUN_NAME}")

    wandb.config.update({
        "run_name": RUN_NAME,
        "lr": LR,
        "epsilon_decay": EPSILON_DECAY,
        "hidden_size": HIDDEN_SIZE,
        "batch_size": BATCH_SIZE,
        "episodes": EPISODES,
        "train_every": TRAIN_EVERY,
        "epsilon_min": EPSILON_MIN,
        "gamma": GAMMA,
        "target_update_every": TARGET_UPDATE_EVERY,
        "n_actions": N_ACTIONS,
        "constant_penalty": -0.5,
        "algorithm": "Double-DQN",
        "resume_command": f"python -m src.sweep3 {wandb.run.sweep_id}"
    }, allow_val_change=True)

    wandb.run.name = f"phase-3-sweep-{wandb.run.id[:4]}"

    agent = DQNAgent(
        input_size=83,
        hidden_size=HIDDEN_SIZE,
        output_size=N_ACTIONS,
        lr=LR,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        epsilon_min=EPSILON_MIN,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
    )

    dummy = torch.zeros(BATCH_SIZE, 83).to(agent.device)
    with torch.no_grad():
        agent.q_network(dummy)
        agent.target_network(dummy)
    print("Network warmed up!")

    os.makedirs(f"experiments/runs/{RUN_NAME}", exist_ok=True)

    recent_rewards = []
    x = 0

    try:
        for x in range(EPISODES):
            obs, info = env.reset()
            done = False
            total_reward = 0
            step_count = 0

            action_counts = [0] * N_ACTIONS
            total_loss = 0
            loss_count = 0
            total_q_value = 0
            grad_norm = 0
            max_speed = 0
            total_speed = 0

            while not done:
                state = process_observation(obs)

                current_speed = obs[0][0]
                max_speed = max(max_speed, current_speed)
                total_speed += current_speed

                action = agent.select_action(state)
                action_counts[action] += 1

                next_obs, reward, terminated, truncated, info = env.step(ACTIONS[action])
                next_state = process_observation(next_obs)
                done = terminated or truncated
                total_reward += reward
                step_count += 1

                agent.store_transition(state, action, reward, next_state, done)
                if step_count % TRAIN_EVERY == 0:
                    result = agent.train()
                    if result[0] is not None:
                        loss, q_val, grad_norm = result
                        total_loss += loss
                        total_q_value += q_val
                        loss_count += 1

                obs = next_obs

            agent.decay_epsilon()

            if x % TARGET_UPDATE_EVERY == 0:
                agent.update_target_network()

            if x % SAVE_EVERY == 0:
                agent.save(f"experiments/runs/{RUN_NAME}/dqn_episode_{x}.pt")

            recent_rewards.append(total_reward)
            if len(recent_rewards) > 50:
                recent_rewards.pop(0)

            avg_loss = total_loss / loss_count if loss_count > 0 else 0
            avg_q = total_q_value / loss_count if loss_count > 0 else 0
            avg_speed = total_speed / step_count if step_count > 0 else 0

            # Build action log dynamically based on action space
            action_log = {f"action_{i}": action_counts[i] / step_count 
                         for i in range(N_ACTIONS)}

            wandb.log({
                "episode": x,
                "total_reward": total_reward,
                "avg_recent_reward": sum(recent_rewards) / len(recent_rewards),
                "epsilon": agent.epsilon,
                "buffer_size": len(agent.buffer),
                "episode_length": step_count,
                "loss": avg_loss,
                "avg_q_value": avg_q,
                "grad_norm": grad_norm,
                "max_speed": max_speed,
                "avg_speed": avg_speed,
                "train_steps": loss_count,
                "buffer_utilization": len(agent.buffer) / BUFFER_SIZE,
                **action_log,
            }, step=x)

            print(f"Episode {x} | Reward: {total_reward:.2f} | Avg: {sum(recent_rewards)/len(recent_rewards):.2f} | Epsilon: {agent.epsilon:.3f} | Steps: {step_count} | Loss: {avg_loss:.4f} | Q: {avg_q:.4f} | Actions: {[f'{c/step_count:.2f}' for c in action_counts]}")

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
    count=2
)