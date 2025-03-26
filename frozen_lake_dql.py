import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
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
class FrozenLakeDQL():
    # Adjustable hyperparameters
    learning_rate_a = 0.001 # learning rate (alpha)
    learning_rate_b = 0.9 # discount rate (gamma)
    network_sync_rate = 10 # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 10000 # size of the replay memory
    minibatch_size = 32 # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.MSELoss() # loss function - Mean Squared Error but could be swapped with something else
    optimizer = None # optimizer - to be initialized later

    ACTIONS = ['L', 'D', 'R', 'U'] # action space (left, down, right, up)

    # Train the FrozenLake environment
    def train(self, episodes, render=False, is_slippery=False):
        # Create FrozenLake instance
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human' if render else None)
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        epsilon = 1 # exploration rate - at forst we take 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network - with the number of nodes in the hidden layer adjustable
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)

        # Make the target and policy networks the same (copy weights/biases from one to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('Policy (random, before training):')
        self.print_dqn(policy_dqn)

        # Policy network optimizer - Adam optimizer can be swapped with something else
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards per episode - initially zeros
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number steps taken - usesd for syncing policy to the target network
        step_count = 0

        for i in range(episodes):
            state = env.reset()[0]
            terminated = False # when the agent falls in hole or has reached goal
            truncated = False # when the agent has taken too many steps - 200 in this case

            # Have the agent move until it falls in a hole or reaches the goal, or takes too many steps
            while not (terminated or truncated):

                # Select action based on epsilon-greedy policy
                if random.random() < epsilon:
                    # act randomly
                    action = env.action_space.sample() # actions - 0=left, 1=down, 2=right, 3=up
                else:
                    # act greedily - but this is a prediction so no training gradient calculations
                    with torch.no_grad():
                        # turn the state into a vector, and every output value corresponds to the Q-value of the action
                        q_values = policy_dqn(self.state_to_dqn_input(state, num_states))
                        action = q_values.argmax().item()

                # Execute action
                new_state,reward,terminated,truncated,_ = env.step(action)

                # Save experience in replay memory
                memory.append((state, action, new_state, reward, terminated))

                # Move to next state
                state = new_state

                # Increment steps
                step_count += 1

            # Keep track of the rewards collected per episode
            if reward == 1: # Did we reach the goal?
                rewards_per_episode[i] = 1

            # Check if enough experience has been collected and if at least 1 reward has been collected (otherwise we wouldn't care to memorize the sequence of actions and results because they led nowhere helpful)
            if len(memory.memory) > self.minibatch_size and np.sum(rewards_per_episode) > 0:
                mini_batch = memory.sample(self.minibatch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Decay epsilon - we want to explore less as we learn more
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count % self.network_sync_rate == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
            
        # Close environment
        env.close()
        
        # Save policy
        torch.save(policy_dqn.state_dict(), 'frozen_lake_dqn.pt')

        # Create new graph
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):x+1])
        plt.subplot(121) # Plot on 1 row, 2 columns, and this is the first plot
        plt.plot(sum_rewards)

        # Plot epsilon decay
        plt.subplot(122) # Plot on 1 row, 2 columns, and this is the second plot
        plt.plot(epsilon_history)

        # Save plots
        plt.savefig('frozen_lake_dqn.png')

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

    # Run frozen lake environment with trained policy
    def test(self, episodes, is_slippery=False):
        # Create FrozenLake instance
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human')
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load('frozen_lake_dqn.pt'))
        policy_dqn.eval() # set model to evaluation mode since not learning

        print('Policy (trained):')
        self.print_dqn(policy_dqn)

        for i in range(episodes):
            state = env.reset()[0] # Initialize to state 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                with torch.no_grad():
                    # Get Q-values for each action and pick the best one - we are completely greedy here since we are testing
                    q_values = policy_dqn(self.state_to_dqn_input(state, num_states))
                    action = q_values.argmax().item()

                new_state, _, terminated, truncated, _ = env.step(action)
                state = new_state
        
        env.close()

    def print_dqn(self, dqn):
        for name, param in dqn.named_parameters():
            print(name, param.data)
        print()


# Main
if __name__ == '__main__':
    frozen_lake_dql = FrozenLakeDQL()
    is_slippery = False
    frozen_lake_dql.train(episodes=1000, is_slippery=is_slippery)
    frozen_lake_dql.test(episodes=4, is_slippery=is_slippery)