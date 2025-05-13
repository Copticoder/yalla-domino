from Policy.BasePolicy import BasePolicy


class RulePolicy(BasePolicy):
    def __init__(self, **kwargs):
        super(RulePolicy, self).__init__()

    # Rules
    def play(self):
        card_counter = [0] * 7
        max_p, max_cnt = None, 0
        # Hand card point counter
        for c in self.state['hand']:
            card_counter[c[0]] += 1
            if card_counter[c[0]] > max_cnt:
                max_p = c[0]
                max_cnt = card_counter[c[0]]

            # Double card
            if c[0] == c[1]:
                continue

            card_counter[c[1]] += 1
            if card_counter[c[1]] > max_cnt:
                max_p = c[1]
                max_cnt = card_counter[c[1]]

        act_sort = (-1, -1, -1, -1)
        rule_act = None
        action_spaces = self.validate_actions
        for act in action_spaces:
            act_card, act_direction, act_inverse = act["card"], act["direction"], act["inverse"]
            _board_pieces = self.state['board'].copy()
            if act_direction == 3 and act_inverse == 3:
                _board_pieces.insert(0, act_card)
            elif act_direction == 3 and act_inverse == 4:
                _board_pieces.insert(0, act_card[::-1])
            elif act_direction == 4 and act_inverse == 3:
                _board_pieces.append(act_card)
            elif act_direction == 4 and act_inverse == 4:
                _board_pieces.append(act_card[::-1])

            # After performing this action, are both ends of the board the same?
            if _board_pieces[0][0] == _board_pieces[-1][-1]:
                board_same = 1
            else:
                board_same = 0

            # After performing this action, does one end of the board satisfy the maximum number of points in the hand?
            if _board_pieces[0][0] == max_p or _board_pieces[-1][-1] == max_p:
                max_cnt_type = 1
            else:
                max_cnt_type = 0

            # Is the card a double card?
            card_double_type = 1 if act_card[0] == act_card[1] else 0
            # Sum of card points
            card_sum = sum(act_card)

            # Priority: Are both ends of the board the same? -> Does one end of the board satisfy the maximum number of points in the hand? -> Is the card a double card? -> Sum of card points
            if board_same > act_sort[0]:
                act_sort = (board_same, max_cnt_type, card_double_type, card_sum)
                rule_act = act
            elif board_same == act_sort[0]:
                if max_cnt_type > act_sort[1]:
                    act_sort = (board_same, max_cnt_type, card_double_type, card_sum)
                    rule_act = act
                elif max_cnt_type == act_sort[1]:
                    if card_double_type > act_sort[2]:
                        act_sort = (board_same, max_cnt_type, card_double_type, card_sum)
                        rule_act = act
                    elif card_double_type == act_sort[2]:
                        if card_sum > act_sort[3]:
                            act_sort = (board_same, max_cnt_type, card_double_type, card_sum)
                            rule_act = act
        return rule_act

    def type(self):
        """Policy type"""
        return "Rule"
