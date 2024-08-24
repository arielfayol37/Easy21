"""
Implementation of RL model to play the game of Easy21
"""
from monte_carlo import MCAgent
from random_agent import RandomAgent
from human_agent import HumanAgent
from sarsa_lambda import SarsaAgent, SarsaApproxAgent

import os
import pickle
import numpy as np

import sys
sys.path.append("../")
from game import Game
from utils import print_progress_bar, display_surface, plot_line

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
        self.reward += reward
    
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
    
    def td_update(self, node, next_state, is_terminal):
        """Executes a temporal difference update given a node and the next state"""
        if type(self.agent) == SarsaAgent or type(self.agent) == SarsaApproxAgent:
            self.agent.td_update(node, next_state, is_terminal)
        else:
            pass
        return None

    def get_action(self, state, possible_actions):
        """ Gets the action for this state under the agent's policy"""
        return self.agent.get_action(state, possible_actions)

    def get_new_node(self, state, action, parent_node=None):
        """ Creates a new node object """
        return Node(state=state, action=action, parent_node=parent_node)

    def load_val_func(self):
        """ Loads a value function that has been persistently stored"""
        is_approx = type(self.agent) == SarsaApproxAgent
        if is_approx:
            paths_exists = os.path.exists(self.pickle_file) and os.path.exists(self.weights_file)
        else:
            paths_exists = os.path.exists(self.pickle_file)
        if paths_exists:
            with open(self.pickle_file, 'rb') as file:
                print("Value function successfully loaded")
                self.agent.Q = pickle.load(file)
            if is_approx:
                with open(self.weights_file, 'rb') as file:
                    print("Weights of Value function successfully loaded")
                    self.agent.weights = pickle.load(file) 
        else:
            if is_approx:
                print(f"Loading files {self.pickle_file} or {self.weights_file} not found")
            else:
                print(f"Loading File {self.pickle_file} not found")  
            sys.exit()

    
    def save_val_func(self):
        """ Persistently stores a value function to storage """
        with open(self.pickle_file, 'wb') as file:
            pickle.dump(self.agent.Q, file)
            print("\n")
            print("Value function successfully saved")
        
        if type(self.agent) == SarsaApproxAgent:
            with open(self.weights_file, 'wb') as file:
                pickle.dump(self.agent.weights, file)
                print("\n")
                print("Weights of Value function successfully saved")            

    def learn(self, load=False, num_train_epochs=1000000, save=True, ref_Q=None, comp_func=None):
        """ Agent learns the action value function """
        agent_type = type(self.agent)
        if agent_type == SarsaAgent or agent_type == SarsaApproxAgent:
            if agent_type == SarsaAgent:
                self.pickle_file = "sarsa_lambda_value_function.pickle"
            else:
                self.pickle_file = "sarsa_approx_value_function.pickle"
                self.weights_file = "sarsa_approx_weights.pickle"
            if load:
                self.load_val_func()
            else:
                print(f"{self.name} is learning...")
                y_vals = []
                x_vals = []
                for i in range(num_train_epochs):
                    print_progress_bar(int((i / num_train_epochs) * 100))
                    game = Game(ai=self, stdout=False)
                    last_node = game.play_game()
                    if not ref_Q is None:
                        x_vals.append(i)
                        y_vals.append(comp_func(self.agent.Q, ref_Q))
                if save:
                    self.save_val_func() 
                # plotting mean squared error against episode number
                if len(y_vals) > 0:
                    labels = {"X":"Episode #", 
                              "Y":"Mean Squared Error"}
                    plot_line(x_vals, y_vals, labels, \
                              f"{str(agent_type.__name__)} MSE vs Episode with lambda = {self.agent._lambda}")

        # if Monte Carlo Agent
        elif type(self.agent) == MCAgent or type(self.agent) == RandomAgent:
            # Load file from pickle
            self.pickle_file = "mc_value_function.pickle"
            if load:
                self.load_val_func()
            else:
                # Play games and learn the value functions
                print(f"{self.name} is learning...")
                print(".")
                last_nodes = []
                batch_size = 1 # number of episodes before updating the action value function
                for i in range(num_train_epochs):
                    print_progress_bar(int((i / num_train_epochs) * 100))
                    # print(f"Epoch #{i}")
                    game = Game(ai=self, stdout=False)
                    last_node = game.play_game()
                    last_nodes.append(last_node)
                    if i % batch_size == 0:
                        while len(last_nodes) > 0:
                            ln = last_nodes.pop()
                            self.agent.backtrack_reward(ln)
                if save:
                    self.save_val_func()
        return None

    def eval(self, num_test_epochs=3, stdout=True):
        """ Evaluates the agent by playing multiple games"""
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
                print("\n")
        
        # print("Value function: ")
        # print(self.agent.Q)
        print("\n")
        print(f"Performance of {self.agent.name}: ")
        for k, v in keys.items():
            print(f"{eqs[k]}: {v}")
        print(f"    Percentage win: {int(keys[1]/num_test_epochs * 100)}%")



def get_mse(Q, ref_Q):
    """ 
    Compares two action value functions, by getting the mean squared error for all
    state-action pairs.
    """
    def get_mean_and_std(Q):
        """ Gets the mean and standard deviation used to normalize values"""
        vals = []
        for state, val in Q.items():
            for action in val["actions"]:
                vals.append(Q[state]["actions"][action]["value"])
        return np.mean(vals), np.std(vals)

    def norm_val(val, mean, std):
        """Normalizes a value given the mean and standard deviation of distribution"""
        return (val - mean) / std
    _sum = 0
    count = 0
    # Q_mean, Q_std = get_mean_and_std(Q)
    # ref_Q_mean, ref_Q_std = get_mean_and_std(ref_Q)
    for state, val in ref_Q.items():
        for action in val["actions"]:
            count += 1
            try:
                _sum += (ref_Q[state]["actions"][action]["value"] \
                        - Q[state]["actions"][action]["value"]) ** 2
            except KeyError:
                _sum += ref_Q[state]["actions"][action]["value"] ** 2
    return _sum / count

def get_sarsa_stats(is_approx=False):
    # Evaluating Sarsa for different lambdas
    lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    errors = []
    for l in lambdas:
        s_agent = SarsaAgent(_lambda=l) if not is_approx else SarsaApproxAgent(_lambda=l)
        s_learner = Learner(s_agent)
        if l in [0, 1]:
            s_learner.learn(load=False, num_train_epochs=1000, save=False, ref_Q=monte_carlo_agent.Q, comp_func=get_mse)
        else:
            s_learner.learn(load=False, num_train_epochs=1000, save=False)
        errors.append(get_mse(s_agent.Q, monte_carlo_agent.Q))
    plot_line(lambdas, errors, labels={"X":"lambda", \
                                    "Y":"mean squared error"}, \
                                        title=f"{type(s_agent).__name__} mean squared error vs lambda value")

    print("Mean Squared Errors at different lambdas: ",errors)


def plot_value_function(Q, title):
    """
    Plots value function in a 3D surface.
    """
    val_func = Q
    table = {}
    d_set, p_set = set(), set()
    for state, val in val_func.items():
        state = eval(state)
        try:
            value = max(val["actions"]["0"]["value"], val["actions"]["1"]["value"])
        except KeyError:
            if "0" in val["actions"]:
                value = val["actions"]["0"]["value"]
            else:
                value = val["actions"]["1"]["value"]
        table[str(state)] = value
        d_show, p_sum = state 
        d_set.add(d_show)
        p_set.add(p_sum)
    d_set, p_set = sorted(list(d_set)), sorted(list(p_set))
    print(title, len(table))
    def func(x, y):
        assert x.shape == y.shape
        def f(i, j):
            key = str([i, j])
            return table[key]

        vectorized_f = np.vectorize(f)
        return vectorized_f(x, y)
    labels = {"X":"dealer showing", "Y":"player's sum", "Z":"Value"}
    display_surface(d_set, p_set, func, labels, title)


if __name__ == "__main__":

    sarsa_approx_agent = SarsaApproxAgent(_lambda=0.3)
    sarsa_approx_learner = Learner(sarsa_approx_agent)
    sarsa_approx_learner.learn(load=True, num_train_epochs=1000000)
    # print(sarsa_approx_agent.Q)
    # print(sarsa_approx_agent.weights)
    # sys.exit()

    monte_carlo_agent = MCAgent()
    mc_learner = Learner(monte_carlo_agent)
    mc_learner.learn(load=True)

    sarsa_agent = SarsaAgent(_lambda=0.3)
    sarsa_learner = Learner(sarsa_agent)
    sarsa_learner.learn(load=True, num_train_epochs=1000000)
    random_agent = RandomAgent()
    r_learner = Learner(random_agent)

    num_games = 100000
    for z in range(1):
        mc_learner.eval(num_games, False)
        sarsa_learner.eval(num_games, False)
        sarsa_approx_learner.eval(num_games, False)
        r_learner.eval(num_games, False)
        print("\n\n\n")
    

    get_sarsa_stats(is_approx=False)
    get_sarsa_stats(is_approx=True)

    plot_value_function(mc_learner.agent.Q, title="Monte Carlo Action Value Function")
    plot_value_function(sarsa_learner.agent.Q, title="Sarsa Action Value Function")
    plot_value_function(sarsa_approx_learner.agent.Q, \
                        title="Sarsa Linear Approximation Value Function")
    
    human_plays = input("Do you wanna play ? Yes(Y) or no(N)")
    if human_plays.lower() in ["y", "yes"]:
        human_agent = HumanAgent(assist_agent=sarsa_agent)
        h_learner = Learner(human_agent)
        h_learner.eval(100, False)
