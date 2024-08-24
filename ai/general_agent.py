import random
class Agent:
    def __init__(self, name):
        self.n_init = 100
        self.name = name
        self.Q = {} # action value function (look-up table)
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

    def get_action(self, state, possible_actions=["1", "0"], game_mode=False):
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