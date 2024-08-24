"""
Python3 implementation of the Easy21 game environment
"""
import random

terminal_state = [0, 0]

class Deck:
    def __init__(self):
        pass
    
    def draw_card(self, is_black=False):
        value = random.randint(1, 10)
        if is_black:
            return value
        if random.random() < 0.34: return -1 * value
        else: return value
        

class Player:
    def __init__(self, name, deck):
        self.name = name
        self.deck = deck
        self.total = self.deck.draw_card(is_black=True)
        self.busted = False

    def hit(self):
        self.total += self.deck.draw_card()
        self.update_status()
    
    def stick(self):
        # self.update_status()
        pass

    def update_status(self):
        self.busted = self.total < 1 or self.total > 21

    
    def dealer_play(self, player_score):
        assert self.name == "dealer"
        while 0 < self.total < 17:
            self.hit()
        if self.busted: return 1
        
        if self.total == player_score: return 0
        elif self.total > player_score: return -1
        else: return 1


class Game:
    def __init__(self, ai=None, stdout=False):
        deck = Deck()
        self.player = Player("player", deck)
        self.dealer = Player("dealer", deck)
        self.ai = ai
        self.stdout = stdout

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


    def play_game(self):
        state, r = [self.dealer.total, self.player.total], 0
        if self.stdout: 
            print("Initial state:")
            print("Player's hand:", self.player.total,\
                   "Dealer's hand:", self.dealer.total)
        node = None
        while True:
            if state == terminal_state: # terminal state
                if self.stdout: 
                    print(f"Final score: {r}")
                break
            if self.stdout: 
                print(f"Player's hand: {self.player.total}")
            possible_actions = self.get_possible_actions(state)
            action = self.ai.get_action(state, possible_actions) if not self.ai is None else self.get_human_input() 
            node = self.ai.get_new_node(state, action, node)
            state, r = self.step(state, action)
            node.add_reward(r)
            self.ai.td_update(node, state, state==terminal_state)
        assert node != None
        return node

    def step(self, state, action):
        d_first, p_sum = state
        reward = 0 

        if int(action) == 1: # hit
            self.player.hit()
            if self.stdout: print("Hit")
        
        if self.player.busted:
            return terminal_state, -1
        
        if int(action) == 0: # stick
            if self.stdout: print("Stick")
            # Dealer takes turns
            reward = self.dealer.dealer_play(player_score=self.player.total)
            if self.stdout: 
                print(f"Dealer's final hand: {self.dealer.total}")
            return terminal_state, reward
        
        else:
            return [d_first, self.player.total], reward
        
    def get_possible_actions(self, state):
        if state == terminal_state:
            return [None]
        else:
            return ["1", "0"]
if __name__ == "__main__":
    game = Game()
    last_node = game.play_game()
