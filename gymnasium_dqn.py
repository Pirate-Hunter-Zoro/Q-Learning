from abc import ABC
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define model - just a feedforward neural network
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions, convolution=False):
        super().__init__()
        self.convolution = convolution
        
        # Define network layers
        if convolution:
            # Calculate the size of the flattened features
            input_channels = 3 # number of channels in the image (RGB)
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=2)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            with torch.no_grad():
                # Source: https://github.com/KianAnd19/NN-car-racing/blob/main/network.py
                sample_input = torch.zeros(1, input_channels, 96, 96)
                x = self.pool(F.relu(self.conv2(F.relu(self.conv1(sample_input)))))
                self.fc_input_dim = x.numel() // x.size(0)
                self.fc0 = nn.Linear(self.fc_input_dim, 1000)
        self.fc1 = nn.Linear(in_states, h1_nodes) # first fully connected layer
        self.fc2 = nn.Linear(h1_nodes, h1_nodes) # second fully connected layer
        self.fc3 = nn.Linear(h1_nodes, h1_nodes) # third fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions) # output layer
                
    def forward(self, x):
        if self.convolution:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x)) # apply rectified linear unit (ReLU) activation function
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x) # no activation function on the output layer
        return x
    

# Define replay buffer
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    def __len__(self):
        return len(self.memory)
    

# Frozen Lake Deep Q-Learning class
class DQL(ABC):
    # Neural Network
    loss_fn = nn.MSELoss() # loss function - Mean Squared Error but could be swapped with something else
    optimizer = None # optimizer - to be initialized later

    def __init__(self, learning_rate_a, learning_rate_b, replay_memory_size, minibatch_size, network_sync_rate):
        self.learning_rate_a = learning_rate_a
        self.learning_rate_b = learning_rate_b
        self.replay_memory_size = replay_memory_size
        self.replay_memory = ReplayMemory(replay_memory_size) # replay memory - stores transitions
        self.minibatch_size = minibatch_size # size of the training data set sampled from the replay memory
        self.network_sync_rate = network_sync_rate # number of steps the agent takes before syncing the policy and target network
        self.epsilon = 1 # exploration rate - at first we take 100% random actions
        self.policy_dqn = None
        self.target_dqn = None

    def state_to_dqn_input(self, state):
        pass
    
    def optimize(self, mini_batch):
        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:
            if terminated:
                # Agent either reached goal or fell into hole (in which case reward is 0)
                # When in terminated state, target q-value is just the reward
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value
                with torch.no_grad():
                    # This is the immediate reward plus the discounted future reward for the next state (the best we could do at said next state)
                    target = torch.FloatTensor(
                        reward + self.learning_rate_b * self.target_dqn(self.state_to_dqn_input(new_state)).max()
                    )
            
            # Get current set of Q-values
            current_q = self.policy_dqn(self.state_to_dqn_input(state))
            current_q_list.append(current_q)

            # Get the target set of Q-values
            target_q = self.target_dqn(self.state_to_dqn_input(state))
            # Update the Q-value for the action taken
            target_q[action] = target
            target_q_list.append(target_q)

        # Now we know what the Q-values "should" be, and what they are
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Now we take a step towards optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # Train the environment
    def train(self, episodes, render=False, is_slippery=False):
        pass

    # Run frozen lake environment with trained policy
    def test(self, episodes):
        pass

    def print_dqn(self, dqn):
        for name, param in dqn.named_parameters():
            print(name, param.data)
        print()