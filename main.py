from maze import Maze
from qlearning import DQNAgent
import numpy as np
import random
import torch
from tqdm import tqdm
import time

MODEL_NAME = "third-" + str(time.time())

open("models/" + MODEL_NAME + ".pt", "w")

# torch.autograd.set_detect_anomaly(True)

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# Environment settings
EPISODES = 5000
SIZE = (10, 10)

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 100  # episodes
SHOW_PREVIEW = True

env = Maze(SIZE[0], SIZE[1])
# env.show_maze()

agent = DQNAgent(size=SIZE)

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, 4)

        new_state, reward, done = env.step(action, step)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        # env.show_maze()
        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.show_maze()
            torch.save(agent.model.state_dict(), f'models/{MODEL_NAME}.pt')

        # Every step we update replay memory and train main network
        agent.replay_memory.append((current_state, action, reward, new_state, done))

        agent.train(done, step)

        current_state = new_state
        step += 1

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

torch.save(agent.model.state_dict(), f'models/{MODEL_NAME}.pt')