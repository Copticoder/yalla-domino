from Policy.BasePolicy import BasePolicy
from env.utils import domino_critics


class HumanPolicy(BasePolicy):
    def __init__(self, **kwargs):
        super(HumanPolicy, self).__init__()

    # Manual input
    def play(self, **kwargs):
        # Input legality judgment
        def human_input_critics(inp, hand_pieces):
            try:
                inp_card, inp_direction = inp.split("@")
                inp_card = eval(inp_card)
                if isinstance(inp_card, list) and len(inp_card) == 2 and inp_direction in ('3', '4') and \
                        inp_card in hand_pieces:
                    return True
                else:
                    print("Check input")
                    return False
            except Exception as e:
                print(e)
                return False

        if len(self.validate_actions) > 1:
            while True:
                print("Choose the card you want to play and its position: 3 for left, 4 for right. Input: card@position, e.g., [3, 3]@3")
                user_inp = input()
                if not human_input_critics(user_inp, self.state['hand']):
                    continue
                act_card, act_direction = user_inp.split("@")
                card_inverse = domino_critics(now_board=self.state['board'],
                                              play_card=eval(act_card),
                                              play_direction=int(act_direction))
                if card_inverse != 0:
                    human_act = {
                        "card": eval(act_card),
                        "direction": int(act_direction),
                        "inverse": card_inverse
                    }
                    return human_act
        else:
            return self.validate_actions[0]

    def type(self):
        """Policy type"""
        return "human"
