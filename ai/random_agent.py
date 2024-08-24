from general_agent import *

class RandomAgent(Agent):
    def __init__(self, name="Random Agent"):
        super().__init__(name)
    
    def get_action(self, state, possible_actions):
        return str(random.randint(0, 1))