from collections import deque
from torch_model import TorchModel
import numpy as np
import random
import torch

REPLAY_MEMORY_SIZE = 10_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
DISCOUNT = 0.9
UPDATE_TARGET_EVERY = 5


class DQNAgent:
    def __init__(self, size=(50, 50), load=False):

        # Main model
        self.model = TorchModel()

        if load is not False:
            self.model.load_state_dict(torch.load(load))
            self.model.eval()

        self.size = size

        # Target network
        self.target_model = TorchModel()
        self.target_model.load_state_dict(self.model.state_dict())

        # An array with last n steps for training
        self.replay_memory = deque(
            maxlen=REPLAY_MEMORY_SIZE)  # [(observation space, action, reward, new observation space, done)]

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = torch.Tensor([transition[0] for transition in minibatch]).view(-1, 1, self.size[0], self.size[1])
        current_qs_list = self.model(current_states).cpu().detach().numpy()

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = torch.Tensor([transition[3] for transition in minibatch]).view(-1, 1, self.size[0], self.size[1])
        future_qs_list = self.target_model(new_current_states).cpu().detach().numpy()

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        inps = torch.Tensor(X).view(-1, 1, self.size[0], self.size[1])
        outs = torch.Tensor(y)

        self.model.batch_train(inps, outs, batch_size=MINIBATCH_SIZE, epochs=1)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0

    def get_qs(self, state):
        with torch.no_grad():
            return self.model(torch.tensor(state).view(-1, 1, self.size[0], self.size[1])).cpu().detach().numpy()
