from general_agent import Agent
import sys 
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
        """ Add a constant(1) to the eligibity of a state-action pair"""
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
    
class SarsaApproxAgent(SarsaAgent):
    def __init__(self, name="Sarsa Linear Approximation Agent", _lambda=0.7, feature_size=36):
        super().__init__(name, _lambda)
        self.feature_size = feature_size
        self.weights = [0 for i in range(feature_size)]
        self.lr = 0.01 # learning rate (step size)
        self.e = 0.05

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
                # alpha = 1 / self.Q[state]["actions"][action]["count"] # decaying step size
                alpha = self.lr # constant step size
                additive = alpha * td_error * self.e_trace[state][action]
                vector_s_a = self.encode_sa(state, action)
                gradient = self.scale_vector(additive, vector_s_a)
                self.update_weights(gradient)
                value = self.dot_product(vector_s_a, self.weights)
                self.Q[state]["actions"][action]["value"] = value
                self.e_trace[state][action] = self.e_discount * self._lambda * \
                                              self.e_trace[state][action]
                seen.add(seen_key)

            node = node.parent

    def get_estm(self, state, action=None):
        """
        Give the current estimate of the value of a state-action pair.
        If action not given, sample action according to our policy.
        """
        sample_action = self.get_action(state) if action is None else action
        state = str(state)
        vector_s_a = self.encode_sa(state, sample_action)
        value = self.dot_product(vector_s_a, self.weights)
        if state in self.Q:
            if sample_action in self.Q[state]["actions"]:
                pass
            else:
                self.Q[state]["actions"][sample_action] = {"count":0, "value":value}
        else:
            self.Q[state] = {"count":0, "actions":{sample_action: \
                                                   {"count":0, "value":value}}}
        return self.Q[state]["actions"][sample_action]["value"]

    def encode_sa(self, state, action):
        """Encodes state action pair into a vector"""
        dealer_ranges = [[1, 4], [4, 7], [7, 10]]
        player_ranges = [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]
        action_ranges = ["1", "0"]

        def get_idx(val, ranges):
            for idx, r in enumerate(ranges):
                if r[0] <= val <= r[1]:
                    return idx
            raise ValueError("Found no idx")

        state = eval(state)
        dealer, player = state
        i = get_idx(dealer, dealer_ranges)
        j = get_idx(player, player_ranges)
        if str(action) == action_ranges[0]:
            k = 0
        elif str(action) == action_ranges[1]:
            k = 1
        else:
            raise ValueError(f"Invalid action: {action}")
        
        one_hot_idx = i * 12 + j * 2 + k
        vector = []
        good = False
        for i in range(self.feature_size):
            if i != one_hot_idx:
                vector.append(0)
            else:
                vector.append(1)
                good = True
        assert good == True, "At least one dimension must be equal to 1"
        return vector 

    def scale_vector(self, c, x):
        return [c * i for i in x]

    def elem_wise_product(self, x, w):
        """ Element-wise multiplication of two vectors"""
        assert len(x) == len(w)
        result = []
        for a, b in zip(x, w):
            result.append(a *b)
        return result 
    
    def dot_product(self, x, w):
        """Dot product between two vectors x and w"""
        assert len(x) == len(w)
        _sum = 0
        for a, b in zip(x, w):
            _sum += a * b
        return _sum
    
    def update_weights(self, gradient):
        """ Updates the weights of action value approximator"""
        assert len(gradient) == len(self.weights)
        for i in range(len(self.weights)):
            self.weights[i] += gradient[i]
        
        

    