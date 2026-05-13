import random
import wandb
from src.utils.environment import make_env
from src.utils.environment import process_observation
from src.dqn.agent import DQNAgent

episodes = 500

run = wandb.init(
    entity="hhs-autonomous-systems",
    project="trackmania-rl",
    name="DQN Agent",
    config={
        "policy": "DQN Agent",
        "episodes": episodes,
    }
)

agent = DQNAgent(
    input_size=83,
    hidden_size=64,
    output_size=4,
    lr=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.05,
    buffer_size=100000,
    batch_size=64
)

env = make_env()

ACTIONS = [
    [1, 0, 0],  # forward
    [1, 0, 1],  # forward + left
    [1, 0, -1], # forward + right
    [0, 1, 0],  # brake
]

for x in range(episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        state = process_observation(obs)

        # 2. select action using agent (epsilon-greedy)
        action = agent.select_action(state)

        # 3. step the environment
        next_obs, reward, terminated, truncated, info = env.step(ACTIONS[action])
        next_state = process_observation(next_obs)
        done = terminated or truncated
        total_reward += reward
        step_count += 1

        # 4. store transition in replay buffer
        agent.store_transition(state, action, reward, next_state, done)
        # 5. train the agent

        loss = agent.train()
        if loss is not None:  # None when buffer not full yet
            wandb.log({"loss": loss})
        else:
            wandb.log({"loss": 0})

        # 6. update obs

        obs = next_obs
    
    # end of episode:

    agent.decay_epsilon()
    # 7. decay epsilon

    # 8. update target network every N episodes
    if x % 10 == 0:
        agent.update_target_network()
    # 9. log to wandb
    
    wandb.log({
    "episode": x,
    "total_reward": total_reward,
    "epsilon": agent.epsilon,
    "buffer_size": len(agent.buffer),  # useful to see when buffer fills up
    "episode_length": step_count,  # add a step counter in your while loop
    })
    print(f"Episode {x}, Total Reward: {total_reward}")

wandb.finish()