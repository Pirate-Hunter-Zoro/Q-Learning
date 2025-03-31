import torch
from agent import Agent
from car_racing import CarRacingDQL
from gymnasium_dqn import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CarRacingAgent(Agent):

    def __init__(self):
        self.desired_first_layer_size = 1000 # We don't have a certain number of states (all the possible frames is nigh limitless), so we need to set the first layer size to a certain number of nodes for the learning model
        self.h1_nodes = 50 
        self.policy_dqn = DQN(in_states=self.desired_first_layer_size, h1_nodes=self.h1_nodes, out_actions=5, convolution=True)
        self.policy_dqn.load_state_dict(torch.load("carracing_dql.pt"))
        self.car_racing_helper = CarRacingDQL()

    def take_action(self, observations, id=0):
        """Takes in a single observation (np.array) and returns
        a discrete action. the id is this agent's number in case 
        you train multiple policies so that your agent class can
        identify which player is taking an action"""
        # Convert the observation to a tensor and preprocess it
        state = self.car_racing_helper.state_to_dqn_input(observations).to(device=device)
        return self.policy_dqn(state).argmax().item()

    def save(self, checkpoint_path):
        """Given a path such as './competition_models/timAgent/'
        we want to create a folder with that name in that path
        and save our model"""

        print("Save not implemeted")

    def load(self, checkpoint_path):
        """Given a path such as './competition_models/timAgent/'
        we want to load our model from that folder"""

        print("Load not implemented")