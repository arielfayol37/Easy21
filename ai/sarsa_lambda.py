from general_agent import Agent

class SarsaAgent(Agent):
    def __init__(self, name="Sarsa Agent", _lambda=0.7):
        super().__init__(name=name)
        self._lambda = _lambda
        self.e_trace = {}
        self.e_discount = 0.8

    def td_update(self, node, next_state, is_terminal):
        """
        Performing a sarsa lambda update given a state-action pair with 
        the actual reward received and the next state the agent will be at.
        """
        state = node.state_repr()
        action = node.action_repr()
        reward = node.reward
        prev_q_val = self.get_estm(state, action)
        
        # Update stats
        self.Q[state]["count"] += 1
        self.Q[state]["actions"][action]["count"] += 1
        self.update_eligibility(state, action)

        # Update value function
        if is_terminal:
            td_error = reward - prev_q_val
        else:
        
            td_error = (reward + self._lambda * self.get_estm(next_state) \
                                - prev_q_val)
        seen = set()
        while node is not None:
            state = node.state_repr()
            action = node.action_repr()
            seen_key = state + "_" + action
            
            if not seen_key in seen:
                alpha = 1 / self.Q[state]["actions"][action]["count"]
                additive = alpha * td_error * self.e_trace[state][action]
                self.Q[state]["actions"][action]["value"] += additive
                self.e_trace[state][action] = self.e_discount * self._lambda * \
                                              self.e_trace[state][action]
                seen.add(seen_key)

            node = node.parent

    def update_eligibility(self, state, action):
        if state in self.e_trace:
            if action in self.e_trace[state]:
                self.e_trace[state][action] += 1
            else:
                self.e_trace[state][action] = 1
        else:
            self.e_trace[state] = {action: 1}

    def get_estm(self, state, action=None):
        """
        Give the current estimate of the value of a state-action pair.
        If action not given, sample action according to our policy.
        """
        sample_action = self.get_action(state) if action is None else action
        init_guess = 0
        state = str(state)
        if state in self.Q:
            if sample_action in self.Q[state]["actions"]:
                pass
            else:
                self.Q[state]["actions"][sample_action] = {"count":0, "value":init_guess}
        else:
            self.Q[state] = {"count":0, "actions":{sample_action: \
                                                   {"count":0, "value":init_guess}}}
        return self.Q[state]["actions"][sample_action]["value"]
    