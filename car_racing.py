import random
from scipy.special import softmax
import torch
from gymnasium_dqn import DQL, DQN, ReplayMemory
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class CarRacingDQL(DQL):
    # Hyperparameters
    def __init__(self, learning_rate_a=0.1, learning_rate_b=0.9, replay_memory_size=100000, minibatch_size=32, network_sync_rate=50000, num_divisions=20, render=False):
        super(CarRacingDQL, self).__init__(learning_rate_a, learning_rate_b, replay_memory_size, minibatch_size, network_sync_rate)
        self.minibatch_size=minibatch_size
        self.num_divisions=num_divisions
        # Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`
        self.env = gym.make("CarRacing-v3", render_mode="human" if render else None, continuous=False) # Discrete actions: 0=left,1=idle,2=right,3=gaz,4=brake
        self.desired_first_layer_size = 32 # We don't have a certain number of states (all the possible frames is nigh limitless), so we need to set the first layer size to a certain number of nodes for the learning model
        self.h1_nodes = 50 # how many nodes in the first hidden layer of the learning model

    def state_to_dqn_input(self, state):
        state = state/255.0 # just normalize the image pixel values
        # Convert to torch tensor if it's not already
        state = np.expand_dims(state, axis=0)
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        return state

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

        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0
        goal_reached=False
        best_rewards=float('-inf')
            
        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal

            rewards = 0

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and rewards>-1000):

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = np.random.choice(np.arange(num_actions), p=self.action_distribution) # actions: 0=left,1=idle,2=right
                else:
                    # select best action            
                    with torch.no_grad():
                        action = self.policy_dqn(self.state_to_dqn_input(state)).argmax().item()

                # Execute action
                new_state,reward,terminated,_,_ = env.step(action)

                # Accumulate reward
                rewards += reward

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated)) 

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count+=1

            # Keep track of the rewards collected per episode.
            rewards_per_episode.append(rewards)
            if(terminated):
                goal_reached = True

            # Graph training progress
            if(i!=0 and i%1000==0):
                print(f'Episode {i} Epsilon {epsilon}')
                                        
                self.plot_progress(rewards_per_episode, epsilon_history)
            
            if rewards>best_rewards:
                best_rewards = rewards
                print(f'Best rewards so far: {best_rewards}')
                # Save policy
                torch.save(self.policy_dqn.state_dict(), f"carracing_dql.pt")

            # Check if enough experience has been collected
            if len(memory)>self.minibatch_size and goal_reached:
                self.action_distribution = self.generate_action_distribution()

                mini_batch = memory.sample(self.minibatch_size)
                self.optimize(mini_batch)        

                # Decay epsilon
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                    step_count=0                    

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
        for _ in range(episodes):
            while not terminated:
                with torch.no_grad():
                    # Get the action from the policy
                    action = self.policy_dqn(self.state_to_dqn_input(state)).argmax().item()
                    # Take the action
                state, _, terminated, _, _ = env.step(action)

        # Close environment
        env.close()