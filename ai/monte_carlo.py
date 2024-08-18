import random

class MCagent:
    def __init__(self, name="Monte Carlo Agent"):
        self.name = name
        self.n_init = 100
        self.Q = {} # state-action pair value table
        # sample self.Q = {"state_1":{"count": 234, "actions":{"action_1":{"value": 0.9, "count": 23}}}

    def get_optimal_action(self, state, possible_actions):
        assert len(possible_actions) > 0
        best = possible_actions[0]
        max_val = -1 * float("inf")
        for action in possible_actions:
            if action in self.Q[state]["actions"]:
                val = self.Q[state]["actions"][action]["value"]
                if val >= max_val:
                    max_val = val
                    best = action
        return best

    def get_action(self, state, possible_actions, game_mode=False):
        """
        The Agent's policy: select the action with best value with a probability of 1 - e, 
                            select another random action with probability of e.
        """
        state = str(state)
        if game_mode:
            if state in self.Q:
                return self.get_optimal_action(state, possible_actions)
            else:
                return str(random.randint(0, 1))
            
        if state in self.Q:
            optimal_action = self.get_optimal_action(state, possible_actions)
            e = self.n_init / (self.n_init + self.Q[state]["count"])
            # print(e)
            if random.random() < e: # explore
                assert len(possible_actions) > 1, \
                    f"there is no exploration possible in the following set of actions: {possible_actions}"
                while True:
                    r = random.randint(0, len(possible_actions) - 1)
                    if possible_actions[r] != optimal_action:
                        chosen_action = possible_actions[r]
                        break
            else:
                chosen_action = optimal_action
        else:
            # there are only two possible actions in this game.
            # Hit(1) or Stick(0)
            chosen_action = str(random.randint(0, 1))
        return chosen_action
        
    def backtrack_reward(self, last_node):
        gamma = 1 # Backtracking reward without discount
        node = last_node
        reward = last_node.reward
        while not node is None:
            reward = gamma * reward
            node.add_reward(reward)
        
            # Update the state-action value functions
            state, action = node.state_repr(), node.action_repr()
            node_r = node.reward
            if state in self.Q:
                if action in self.Q[state]["actions"]:
                    self.Q[state]["actions"][action]["count"] += 1
                    alpha = 1 / self.Q[state]["actions"][action]["count"]
                    additive = round(alpha * (node_r - self.Q[state]["actions"][action]["value"]), 3)
                    self.Q[state]["actions"][action]["value"] += additive
                else:
                    self.Q[state]["actions"][action] = {"count":1, "value":node_r}
                self.Q[state]["count"] += 1
            
            else:
                self.Q[state] = {"count":1, "actions":{action: {"count":1, "value":node_r}}}
            
            # Update last node
            last_node = node.parent
            del node
            node = last_node
        return None
    
class RandomAgent(MCagent):
    def __init__(self, name="Random Agent"):
        super().__init__(name)
    
    def get_action(self, state, possible_actions):
        return str(random.randint(0, 1))
    
class HumanAgent(MCagent):
    def __init__(self, name="Human Agent"):
        super().__init__(name)
    
    def get_action(self, state, possible_actions):
        print(f"Player's Hand: {state[1]}, Dealer's Hand: {state[0]}")
        return self.get_human_input()
    
    def get_human_input(self):
        while True:
            try:
                is_hit = int(input("Hit(1) or Stick(0):\n..."))
                if not is_hit in [0, 1]:
                    print("Enter Either 0 or 1 as input")
                    raise ValueError("Invalid Input: expected either a 1 or a 0")
                break
            except Exception as e:
                print(e)
                print("Invalid integer input. Try again\n")
        return is_hit
        