import random
from env.utils import card_map
from env.DominoMonitor import DominoMonitor


class GameStart:
    """
    开始游戏
    """

    def __init__(self):
        # 基础domino
        self.dominos = [[int(card[0]), int(card[2])] for card in card_map.keys()]

    def game_init(self, is_start_round=False):
        # 随机打乱
        new_d = self.dominos.copy()
        while True:
            if is_start_round:
                random.shuffle(new_d)
                # 拆分骨牌：共28张牌
                start_stock_pieces, start_opponent_pieces, start_player_pieces = new_d[:14], new_d[14:21], new_d[21:]
                # 特殊处理：发牌时必发双牌
                p_double = [[x, y] for x, y in start_player_pieces if x == y]
                o_double = [[x, y] for x, y in start_opponent_pieces if x == y]
                if not p_double or not o_double:
                    continue

                max_p_double = max(p_double)
                max_o_double = max(o_double)
                start_game_state = DominoMonitor(
                    player_pieces=start_player_pieces,
                    opponent_pieces=start_opponent_pieces,
                    stock_pieces=start_stock_pieces,
                    board_pieces=[],
                    turn_sign=1 if max_p_double[0] > max_o_double[0] else -1
                )
                # 走规则规定的第一步：更新游戏状态
                start_game_state.act_state_update(
                    {
                        "card": max_p_double if max_p_double[0] > max_o_double[0] else max_o_double,
                        "direction": 3,
                        "inverse": 3
                    }
                )
                return start_game_state.player_pieces, start_game_state.opponent_pieces, \
                       start_game_state.stock_pieces, start_game_state.board_pieces, start_game_state.turn_sign
            else:
                random.shuffle(new_d)
                # 拆分骨牌：共28张牌
                start_stock_pieces, start_opponent_pieces, start_player_pieces = new_d[:14], new_d[14:21], new_d[21:]
                return start_player_pieces, start_opponent_pieces, start_stock_pieces, [], random.choice([1, -1])
