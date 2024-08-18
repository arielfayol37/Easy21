"""
Implementation of RL model to play the game of Easy21
"""
from monte_carlo import MCagent, RandomAgent, HumanAgent
import os
import pickle
import numpy as np

import sys
sys.path.append("../")
from game import Game
from utils import print_progress_bar, display_surface

class Node:
    """
    Representation of a state, action, then new state in an episode
    """
    def __init__(self, state, action, parent_node=None):
        self.parent = parent_node
        self.state = state
        self.action = action
        self.reward = 0
    
    def add_reward(self, reward):
        self.reward += round(reward, 3)
    
    def state_repr(self):
        """ Gets the string representation of the state"""
        return str(self.state)
    
    def action_repr(self):
        """Gets the string representation of the action"""
        return str(self.action)
    
    def __str__(self):
        return str(self.state) + "_" + str(self.action)
    
    def __repr__(self):
        return self.__str__()

class Learner:
    def __init__(self, agent):
        self.agent = agent
        self.name = self.agent.name

    def get_action(self, state, possible_actions):
        return self.agent.get_action(state, possible_actions)

    def get_new_node(self, state, action, parent_node=None):
        return Node(state=state, action=action, parent_node=parent_node)

    def learn(self, load=False, num_train_epochs=10000000):
        # if Monte Carlo Agent
        if type(self.agent) == MCagent or type(self.agent) == RandomAgent:
            # Load file from pickle
            pickle_file = "mc_value_function.pickle"
            if load and os.path.exists(pickle_file):
                with open(pickle_file, 'rb') as file:
                    self.agent.Q = pickle.load(file)
                    print("Value function successfully loaded")
            else:
                if load:
                    print("Loading File not found.")
                # Play games and learn the value functions
                print("Agent is learning...")
                print(".")
                for i in range(num_train_epochs):
                    print_progress_bar(int((i / num_train_epochs) * 100))
                    # print(f"Epoch #{i}")
                    game = Game(ai=self, stdout=False)
                    last_node = game.play_game()
                    self.agent.backtrack_reward(last_node)
                with open(pickle_file, 'wb') as file:
                    pickle.dump(self.agent.Q, file)
                    print("Value function successfully saved")
            # print(self.agent.Q)
        return None

    def eval(self, num_test_epochs=3, stdout=True):
        print("\n\n")
        # Evaluate agent by watching it play
        reward = 0
        keys = {0:0, 1:0, -1:0}
        eqs = {0:"draws", 1:"wins", -1:"losses"}
        for j in range(num_test_epochs):
            game = Game(ai=self, stdout=stdout)
            last_node = game.play_game()
            r = last_node.reward
            if type(self.agent) == HumanAgent: 
                print(f"Game #{j}: You ", eqs[r], "\n")
            reward += last_node.reward
            keys[r] += 1
            if stdout: 
                print("\n\n")
        
        # print("Value function: ")
        # print(self.agent.Q)
        print("\n\n")
        print(f"Performance of {self.agent.name}: ")
        for k, v in keys.items():
            print(f"{eqs[k]}: {v}")
        print(f"    Percentage win: {int(keys[1]/num_test_epochs * 100)}%")


if __name__ == "__main__":
    monte_carlo_agent = MCagent()
    mc_learner = Learner(monte_carlo_agent)
    mc_learner.learn(load=True)
    mc_learner.eval(10000, False)

    random_agent = RandomAgent()
    r_learner = Learner(random_agent)
    r_learner.eval(10000, False)

    # human_agent = HumanAgent()
    # h_learner = Learner(human_agent)
    # h_learner.eval(100, False)

    # Plot value function
    val_func = mc_learner.agent.Q
    table = {}
    d_set, p_set = set(), set()
    for state, val in val_func.items():
        state = eval(state)
        value = max(val["actions"]["0"]["value"], val["actions"]["1"]["value"])
        table[str(state)] = value
        d_show, p_sum = state 
        d_set.add(d_show)
        p_set.add(p_sum)
    d_set, p_set = sorted(list(d_set)), sorted(list(p_set))

    def func(x, y):
        assert x.shape == y.shape
        def f(i, j):
            key = str([i, j])
            return table[key]
        vectorized_f = np.vectorize(f)
        return vectorized_f(x, y)
    labels = {"X":"dealer showing", "Y":"player's sum", "Z":"Value"}
    display_surface(d_set, p_set, func, labels)
