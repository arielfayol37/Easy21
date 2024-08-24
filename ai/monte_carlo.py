from general_agent import *
class MCAgent(Agent):
    def __init__(self, name="Monte Carlo Agent"):
        super().__init__(name=name)
        
    def backtrack_reward(self, last_node):
        """
        Backtracking the rewards (to get the returns for each state-action pair)
        after an episode.
        """
        gamma = 1 # Discount
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
                    alpha = 1 / self.Q[state]["actions"][action]["count"] # decaying alpha but doesn't forget past experience.
                    # alpha = 0.02 # 5% ==> meant for forgetting past experience
                    additive = alpha * (node_r - self.Q[state]["actions"][action]["value"])
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