import random
from scipy.special import softmax
import torch
from gymnasium_dqn import DQL, DQN, ReplayMemory
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

class CarRacingDQL(DQL):
    # Hyperparameters
    def __init__(self, learning_rate_a=1e-4, learning_rate_b=0.9, replay_memory_size=10000, minibatch_size=64, network_sync_rate=50000, num_divisions=20, render=False):
        super(CarRacingDQL, self).__init__(learning_rate_a, learning_rate_b, replay_memory_size, minibatch_size, network_sync_rate)
        self.minibatch_size=minibatch_size
        self.num_divisions=num_divisions
        # Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`
        self.env = gym.make("CarRacing-v3", render_mode="human" if render else None, continuous=False) # Discrete actions: 0=left,1=idle,2=right,3=gaz,4=brake
        self.desired_first_layer_size = 1000 # We don't have a certain number of states (all the possible frames is nigh limitless), so we need to set the first layer size to a certain number of nodes for the learning model
        self.h1_nodes = 50 # how many nodes in the first hidden layer of the learning model

    def state_to_dqn_input(self, state):
        state = state/255.0 # just normalize the image pixel values
        # Convert to torch tensor if it's not already
        state = np.expand_dims(state, axis=0)
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        state = state.permute(0, 3, 1, 2) # change the order of the dimensions to (batch_size, channels, height, width)
        return state.squeeze()

    def generate_action_distribution(self):
        # Generate a random action distribution
        action_distribution = np.random.rand(self.num_actions)
        action_distribution = softmax(action_distribution) # convert to a probability distribution
        return action_distribution

    def train(self, episodes):
        env = self.env
        obs, _ = env.reset()
        state = obs[2]
        num_actions = env.action_space.n # 5 - none, steer left, steer right, gas, break
        self.num_actions = num_actions
        epsilon_end = 0.01 # minimum exploration rate
        epsilon_decay = 0.99 # decay rate for exploration

        epsilon = 1 # exploration rate - at first we take 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network - with the number of nodes in the hidden layer adjustable
        self.policy_dqn = DQN(in_states=self.desired_first_layer_size, h1_nodes=self.h1_nodes, out_actions=num_actions, convolution=True)
        self.target_dqn = DQN(in_states=self.desired_first_layer_size, h1_nodes=self.h1_nodes, out_actions=num_actions, convolution=True)
        self.action_distribution = self.generate_action_distribution()

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        
        # Policy network optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = []

        # List to keep track of epsilon decay
        epsilon_history = []

        best = -np.inf
        # Training loop
        for epoch in range(episodes):
            state = env.reset()[0]
            state = self.state_to_dqn_input(state).to(device)
            done = False
            total_reward = 0

            while not done:
                if np.random.random() < epsilon:
                    if random.random() > 0.5:
                        action = 3
                    else:
                        action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_values = self.policy_dqn(state.unsqueeze(0)).squeeze()
                        action = q_values.cpu().numpy().argmax()

                
                next_state, reward, done, truncated, _ = env.step(action)
                            
                total_reward += reward
                memory.append((state, action, reward, self.state_to_dqn_input(next_state), done))
                
                if len(memory) >= self.replay_memory_size:
                    batch = memory.sample(self.minibatch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)

                    states = torch.stack(states).to(device)
                    actions = torch.FloatTensor(np.array(actions)).to(device)
                    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                    next_states = torch.stack(next_states).to(device)
                    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
                    
                    # Predict Q-values for the current states
                    current_q_values = self.policy_dqn(states)
                    
                    # Predict Q-values for the next states
                    next_q_values = self.target_dqn(next_states)
                    
                    # Calculate target Q-values
                    target_q_values = rewards + (1 - dones) * self.learning_rate_b * next_q_values

                    
                    # Compute the loss between the current Q-values and target Q-values
                    loss = torch.nn.MSELoss()(current_q_values, target_q_values)

                    self.optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.(q_network.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    # scheduler.step()
                
                state = self.state_to_dqn_input(next_state)
                done = done or truncated
                
                if done:
                    rewards_per_episode.append(total_reward)
                    self.plot_progress(rewards_per_episode, epsilon_history)
                    print(f"Epoch {epoch + 1}, Total Reward: {total_reward}, Epsilon: {epsilon}, Best: {best}")
                    
                    if best < total_reward:
                        best = total_reward
                        torch.save(self.policy_dqn.state_dict(), 'carracing_dql.pt')
                    
                    break
                
            if epoch % self.network_sync_rate == 0:
                self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                
            epsilon = max(epsilon_end, epsilon * epsilon_decay) 
            self.plot_progress(rewards_per_episode=rewards_per_episode, epsilon_history=epsilon_history)
            epsilon = max(epsilon_end, epsilon * epsilon_decay)                   

        # Close environment
        env.close()

    def plot_progress(self, rewards_per_episode, epsilon_history):
        # Create new graph
        plt.figure(1)

        # Plot rewards per episode
        plt.subplot(121) # one row, two columns, first graph
        plt.plot(rewards_per_episode)
        # Plot epsilon decay
        plt.subplot(122) # one row, two columns, second graph
        plt.plot(epsilon_history)

        # Save plots
        plt.savefig('carracing_dql.png')

    # Run mountain car environment with trained policy
    def test(self, episodes):
        env = gym.make("CarRacing-v3", render_mode="human")
        state = env.reset()[0]
        num_actions = env.action_space.n

        # Load learned policy
        self.policy_dqn = DQN(in_states=self.desired_first_layer_size, h1_nodes=self.h1_nodes, out_actions=num_actions)
        self.policy_dqn.load_state_dict(torch.load('carracing_dql.pt'))
        self.policy_dqn.eval() # set model to evaluation mode since not learning
        terminated = False
        truncated = False
        for _ in range(episodes):
            while not terminated or truncated:
                with torch.no_grad():
                    # Get the action from the policy
                    action = self.policy_dqn(self.state_to_dqn_input(state)).argmax().item()
                    # Take the action
                state, _, terminated, truncated, _ = env.step(action)

        # Close environment
        env.close()