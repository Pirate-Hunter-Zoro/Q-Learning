import random
import torch
from gymnasium_dqn import DQL, DQN, ReplayMemory
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class MountainCarDQL(DQL):
    # Hyperparameters
    def __init__(self, learning_rate_a=0.001, learning_rate_b=0.9, replay_memory_size=100000, minibatch_size=32, network_sync_rate=50000, num_divisions=20):
        super(MountainCarDQL, self).__init__(learning_rate_a, learning_rate_b, replay_memory_size, minibatch_size, network_sync_rate)
        self.num_divisions=num_divisions
        
    def state_to_dqn_input(self, state):
        # We have bins of position spaces and velocity spaces - we need to convert the state to the index of the bin
        state_p = np.digitize(state[0], self.pos_space) # position
        state_v = np.digitize(state[1], self.vel_space) # velocity
        return torch.FloatTensor([state_p, state_v])

    def train(self, episodes, render=False):
        env = gym.make("MountainCar-v0", render_mode="human" if render else None)
        num_states = env.observation_space.shape[0] # 2 - position and velocity
        num_actions = env.action_space.n # 3 - left, right, none

        # divide the state space into discrete states
        self.pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], self.num_divisions) # Between -1.2 and 0.6
        self.vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], self.num_divisions) # Between -0.07 and 0.07

        epsilon = 1 # exploration rate - at first we take 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network - with the number of nodes in the hidden layer adjustable
        self.policy_dqn = DQN(in_states=num_states, h1_nodes=10, out_actions=num_actions)
        self.target_dqn = DQN(in_states=num_states, h1_nodes=10, out_actions=num_actions)

        # Make target and policy networks the same
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

        # Optimizer which can be changed
        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards per episode - initially zeros
        rewards_per_episode = np.zeros(episodes)
        # List to keep track of epsilon decay
        epsilon_history = []
        # Track number steps taken - used for syncing policy to the target network
        step_count = 0
        goal_reached = False
        best_reward = -200

        for i in range(episodes):
            state = env.reset()[0]
            terminated = False # when the agent falls in hole or has reached goal
            
            rewards = 0
            while (not terminated and rewards>-1000):

                # Select action based on epsilon-greedy policy
                if random.random() < epsilon:
                    # act randomly
                    action = env.action_space.sample()
                else:
                    # act according to greedy policy selecting the best action
                    with torch.no_grad():
                        action = self.policy_dqn(self.state_to_dqn_input(state)).argmax().item()

                # Take the action
                new_state, reward, terminated, _, _ = env.step(action)

                # Add reward to the total reward for this episode
                rewards += reward
                # Store the transition in the replay memory
                memory.append((state, action, new_state, reward, terminated))
                # Update the state
                state = new_state
                # Increment step count
                step_count += 1

            # Store the total reward for this episode
            rewards_per_episode[i] = rewards
            if terminated:
                goal_reached = True

            # Graph training progress
            if not (i % 1000):
                print(f"Episode {i} - Reward: {rewards} - Epsilon: {epsilon:.2f} - Steps: {step_count}")
                self.plot_progress(rewards_per_episode, epsilon_history)

            if rewards > best_reward:
                best_reward = rewards
                print(f"Best reward: {best_reward} - Epsilon: {epsilon:.2f} - Steps: {step_count}")
                # Save the model
                torch.save(self.policy_dqn.state_dict(), 'mountain_car_dqn.pt')

            # If enough experiences in memory, start learning
            if len(memory) > self.minibatch_size and goal_reached: # only worth training if we actually reached the goal at least once
                mini_batch = memory.sample(self.minibatch_size)
                self.optimize(mini_batch)

                # Decay epsilon - we want to explore less as we learn more
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if not (step_count % self.network_sync_rate):
                    # Sync the policy network to the target network
                    self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        
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
        plt.savefig('mountain_car_dqn.png')

    # Run mountain car environment with trained policy
    def test(self, episodes):
        env = gym.make("MountainCar-v0", render_mode="human")
        state = env.reset()[0]
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load learned policy
        self.policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        self.policy_dqn.load_state_dict(torch.load('mountain_car_dqn.pt'))
        self.policy_dqn.eval() # set model to evaluation mode since not learning
        terminated = False
        for _ in range(episodes):
            while not terminated:
                with torch.no_grad():
                    # Get the action from the policy
                    action = self.policy_dqn(self.state_to_dqn_input(state)).argmax().item()
                    # Take the action
                state, _, terminated, _, _ = env.step(action)

        # Close environment
        env.close()