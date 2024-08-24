from general_agent import * 
class HumanAgent(Agent):
    def __init__(self, name="Human Agent", assist_agent=None):
        super().__init__(name)
        self.assist_agent = assist_agent
    
    def get_action(self, state, possible_actions):
        """ Get action from human input (with maybe an assist from an AI agent)"""
        print(f"Player's Hand: {state[1]}, Dealer's Hand: {state[0]}")
        if not self.assist_agent is None:
            print(f"{self.assist_agent.name} recommends:",\
                   self.assist_agent.get_action(state, possible_actions))
        return self.get_human_input()
    
    def get_human_input(self):
        """ Get and validate human input"""
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
        