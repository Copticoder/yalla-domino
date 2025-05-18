from collections import defaultdict


# 游戏环境模拟器
class DominoMonitor:
    def __init__(self, player_pieces, opponent_pieces, stock_pieces, board_pieces, turn_sign):
        self.player_pieces = player_pieces
        self.opponent_pieces = opponent_pieces
        self.stock_pieces = stock_pieces
        self.board_pieces = board_pieces
        self.turn_sign = turn_sign

        # 当前操作对象
        self.agent_pieces = self.player_pieces if self.turn_sign > 0 else self.opponent_pieces

        # 初始化点数计数器
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
        # player没有手牌
        if not self.player_pieces:
            # print("\n游戏结束. 你赢了!")
            # print("你没有手牌了")
            return sum([sum(val) for val in self.opponent_pieces])

        # computer没有手牌
        if not self.opponent_pieces:
            # print("\n游戏结束. 对手赢了!")
            # print("对手没有手牌了")
            return -sum([sum(val) for val in self.player_pieces])

        # 头、尾的点牌已经耗尽
        if self.card_count.get(self.board_pieces[0][0]) == 8 and \
                self.card_count.get(self.board_pieces[-1][-1]) == 8:
            # print("无法继续接牌，比较点数大小")
            # 结算
            p_score = sum([sum(val) for val in self.player_pieces])
            c_score = sum([sum(val) for val in self.opponent_pieces])
            if p_score <= c_score:
                # print("\n游戏结束，点数较小，你赢了!")
                return c_score - p_score
            else:
                # print("\n游戏结束，点数较小，对手赢了!")
                return -(p_score - c_score)

        # 游戏继续
        return None

    # 手牌是否有动作空间
    def _hand_connect_sign(self, now_hand):
        # 待连接点
        key_point = [self.board_pieces[0][0], self.board_pieces[-1][-1]]
        # 手牌中满足出牌要求
        return any([point in key_point for card in now_hand for point in card])

    # 根据出牌动作，更新游戏环境，不判断胜利条件
    def _update_states(self, act):
        act_card, act_direction, act_inverse = act["card"], act["direction"], act["inverse"]
        # 从agent的手牌中删除
        self.agent_pieces.remove(act_card)
        # 判断是否翻转，添加到board
        inverse_card = act_card if act_inverse == 3 else act_card[::-1]
        if act_direction == 3:
            self.board_pieces.insert(0, inverse_card)
        else:
            self.board_pieces.append(inverse_card)
        # 更新点数计数器
        for v in act_card:
            self.card_count[v] += 1

    # 输入动作，并更新游戏环境，判断游戏胜利条件
    def act_state_update(self, act=None):
        # 无动作空间 并且 牌库为空：判断游戏是否结束
        if act is None and not self.stock_pieces:
            # 交换出牌顺序
            self.turn_sign *= -1
            # 当前操作角色手牌
            self.agent_pieces = self.player_pieces if self.turn_sign > 0 else self.opponent_pieces
            return self._win_condition()

        # 有动作空间，根据act，更新游戏状态
        self._update_states(act)

        # 出牌后手牌是否为空
        if not self.agent_pieces:
            return self._win_condition()

        # 交换出牌顺序
        self.turn_sign *= -1
        # 当前操作角色手牌
        self.agent_pieces = self.player_pieces if self.turn_sign > 0 else self.opponent_pieces

        # 手牌无法连接board，并且牌库有牌
        if not self._hand_connect_sign(self.agent_pieces) and self.stock_pieces:
            # 持续发牌直到：能连接 或 牌库为空
            while not self._hand_connect_sign(self.agent_pieces) and self.stock_pieces:
                deal = self.stock_pieces.pop()
                # print("补牌 + 1")
                self.agent_pieces.append(deal)

        # 判断发牌后是否能接牌，不能接牌交换
        if not self._hand_connect_sign(self.agent_pieces):
            self.turn_sign *= -1
            self.agent_pieces = self.player_pieces if self.turn_sign > 0 else self.opponent_pieces

        # 判断游戏是否结束
        return self._win_condition()
