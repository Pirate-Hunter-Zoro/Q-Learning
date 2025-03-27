from abc import ABC
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define model - just a feedforward neural network
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes) # first fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions) # output layer
                
    def forward(self, x):
        x = F.relu(self.fc1(x)) # apply rectified linear unit (ReLU) activation function
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
    

# Frozen Lake Deep Q-Learning class
class DQL(ABC):
    # Adjustable hyperparameters
    learning_rate_a = 0.001 # learning rate (alpha)
    learning_rate_b = 0.9 # discount rate (gamma)
    network_sync_rate = 10 # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 10000 # size of the replay memory
    minibatch_size = 32 # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.MSELoss() # loss function - Mean Squared Error but could be swapped with something else
    optimizer = None # optimizer - to be initialized later

    def state_to_dqn_input(self, state, num_states):
        input_tensor = torch.zeros(num_states)
        # In this case the state just correponds with a number, so we set the value at that index to 1
        input_tensor[state] = 1
        return input_tensor
    
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # Get the number of input nodes - this is the number of states
        num_states = policy_dqn.fc1.in_features
        
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
                        reward + self.learning_rate_b * target_dqn(self.state_to_dqn_input(new_state, num_states)).max()
                    )
            
            # Get current set of Q-values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Get the target set of Q-values
            target_q = target_dqn(self.state_to_dqn_input(state, num_states))
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