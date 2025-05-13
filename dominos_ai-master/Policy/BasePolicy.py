class BasePolicy:
    """Base policy"""

    def __init__(self, **kwargs):
        self.validate_actions = None
        self.state = None
        self.is_start_round = None

    def update_state(self, now_hand, now_board, opponent_num, stock_num, validate_actions, is_start_round):
        """Update current state        """
        self.validate_actions = validate_actions.copy()
        self.state = {
            "hand": now_hand.copy(),
            "board": now_board.copy(),
            "op_num": opponent_num,
            "stock_num": stock_num
        }
        self.is_start_round = is_start_round

    def play(self):
        """Play card"""
        pass

    def type(self):
        """Policy type"""
        return "Base"
