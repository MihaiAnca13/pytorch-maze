from maze import Maze
from qlearning import DQNAgent
import numpy as np
import random
import torch
from tqdm import tqdm

SIZE = (10, 10)
EPISODES = 1

env = Maze(SIZE[0], SIZE[1])

agent = DQNAgent(size=SIZE, load="models/first-1584700524.8551683.pt")

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
        action = np.argmax(agent.get_qs(current_state))
        print(action)

        current_state, reward, done = env.step(action, step, show=True, delay=0)

        episode_reward += reward

        step += 1

    print(f"ep reward: {episode_reward}")