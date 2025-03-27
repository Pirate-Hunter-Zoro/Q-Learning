from gymnasium_dqn import DQL

class MountainCarDQL(DQL):
    def __init__(self):
        super(MountainCarDQL, self).__init__()
        
    def train(self, episodes, render=False):
        pass

    # Run mountain car environment with trained policy
    def test(self, episodes):
        pass