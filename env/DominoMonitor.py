from collections import defaultdict


# Game environment simulator
class DominoMonitor:
    def __init__(self, player_pieces, opponent_pieces, stock_pieces, board_pieces, turn_sign):
        self.player_pieces = player_pieces
        self.opponent_pieces = opponent_pieces
        self.stock_pieces = stock_pieces
        self.board_pieces = board_pieces
        self.turn_sign = turn_sign

        # Current operating object
        self.agent_pieces = self.player_pieces if self.turn_sign > 0 else self.opponent_pieces

        # Initialize point counter
        self.card_count = defaultdict(int)
        for x, y in self.board_pieces:
            self.card_count[x] += 1
            self.card_count[y] += 1

    def get_state(self):
        return {
            'hand': self.player_pieces.copy() if self.turn_sign > 0 else self.opponent_pieces.copy(),
            'board': self.board_pieces.copy(),
            'op_num': len(self.opponent_pieces) if self.turn_sign > 0 else len(self.player_pieces),
            'stock_num': len(self.stock_pieces)
        }

    def _win_condition(self):
        # Player has no hand cards
        if not self.player_pieces:
            # print("\nGame over. You win!")
            # print("You have no hand cards left")
            return sum([sum(val) for val in self.opponent_pieces])

        # Computer has no hand cards
        if not self.opponent_pieces:
            # print("\nGame over. Opponent wins!")
            # print("Opponent has no hand cards left")
            return -sum([sum(val) for val in self.player_pieces])

        # The point cards at the head and tail have been exhausted
        if self.card_count.get(self.board_pieces[0][0]) == 8 and \
                self.card_count.get(self.board_pieces[-1][-1]) == 8:
            # print("Cannot continue to connect cards, compare point sizes")
            # Settlement
            p_score = sum([sum(val) for val in self.player_pieces])
            c_score = sum([sum(val) for val in self.opponent_pieces])
            if p_score <= c_score:
                # print("\nGame over, points are smaller, you win!")
                return c_score - p_score
            else:
                # print("\nGame over, points are smaller, opponent wins!")
                return -(p_score - c_score)

        # Game continues
        return None

    # Whether the hand has action space
    def _hand_connect_sign(self, now_hand):
        # Points to be connected
        key_point = [self.board_pieces[0][0], self.board_pieces[-1][-1]]
        # Hand meets the card playing requirements
        return any([point in key_point for card in now_hand for point in card])

    # Update the game environment according to the card playing action, without judging the victory condition
    def _update_states(self, act):
        act_card, act_direction, act_inverse = act["card"], act["direction"], act["inverse"]
        # Remove from agent's hand
        self.agent_pieces.remove(act_card)
        # Judge whether to flip, add to board
        inverse_card = act_card if act_inverse == 3 else act_card[::-1]
        if act_direction == 3:
            self.board_pieces.insert(0, inverse_card)
        else:
            self.board_pieces.append(inverse_card)
        # Update point counter
        for v in act_card:
            self.card_count[v] += 1

    # Input action, update game environment, and judge game victory conditions
    def act_state_update(self, act=None):
        # No action space and stock is empty: judge whether the game is over
        if act is None and not self.stock_pieces:
            # Exchange turn order
            self.turn_sign *= -1
            # Current operating role hand cards
            self.agent_pieces = self.player_pieces if self.turn_sign > 0 else self.opponent_pieces
            return self._win_condition()

        # There is action space, update the game state according to act
        self._update_states(act)

        # Whether the hand is empty after playing a card
        if not self.agent_pieces:
            return self._win_condition()

        # Exchange turn order
        self.turn_sign *= -1
        # Current operating role hand cards
        self.agent_pieces = self.player_pieces if self.turn_sign > 0 else self.opponent_pieces

        # Hand cannot connect to board, and stock has cards
        if not self._hand_connect_sign(self.agent_pieces) and self.stock_pieces:
            # Continue to deal cards until: can connect or stock is empty
            while not self._hand_connect_sign(self.agent_pieces) and self.stock_pieces:
                deal = self.stock_pieces.pop()
                # print("Draw card + 1")
                self.agent_pieces.append(deal)

        # Judge whether it can be connected after dealing cards, if not, exchange
        if not self._hand_connect_sign(self.agent_pieces):
            self.turn_sign *= -1
            self.agent_pieces = self.player_pieces if self.turn_sign > 0 else self.opponent_pieces

        # Judge whether the game is over
        return self._win_condition()
