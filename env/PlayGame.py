import random
from env.GameStart import GameStart
from env.DominoMonitor import DominoMonitor
from env.utils import get_validate_act, convert_action_to_index


class PlayGame:
    """玩游戏"""

    def __init__(self, Policy1, Policy2, print_details=False, **kwargs):
        # 打印游戏信息
        self.print_details = print_details

        self.Policy1 = Policy1
        self.Policy2 = Policy2

        self.model_action_rank = []

    def get_state_info(self, game_state):
        """
        打印对局信息
        :param game_state:
        """
        if not self.print_details:
            return
        print("-" * 50)
        print("Player1 手牌:",
              len(game_state.player_pieces) if self.Policy2.type() == 'human' else game_state.player_pieces)
        print("Player2 手牌:",
              len(game_state.opponent_pieces) if self.Policy1.type() == 'human' else game_state.opponent_pieces)
        print("Board:", game_state.board_pieces)
        if self.Policy1.type() != 'human' and self.Policy2.type() != 'human':
            print("Stock:", game_state.stock_pieces)
        else:
            print("Stock:", len(game_state.stock_pieces))
        print("-" * 50)

    def get_init_info(self):
        """
        打印对局初始化信息
        :param game_state:
        """
        if not self.print_details:
            return
        print("#" * 25 + "Game Start" + "#" * 25)

    def print_final_info(self, round_win_score):
        """
        打印对局结束信息
        :param:
        """
        if not self.print_details:
            return
        if round_win_score > 0:
            print("本轮游戏结束，Player1赢了～, 得分", round_win_score)
        elif round_win_score < 0:
            print("本轮游戏结束，Player2赢了～, 得分", round_win_score)
        else:
            print("平局.")

        print("#" * 25 + "Game Ended" + "#" * 25)

    def run_game(self):
        """进行一次对局"""
        self.get_init_info()
        # 以 1/4 的概率生成round 1对局
        is_start_round = random.random() > 0.75
        p_pieces, o_pieces, s_pieces, b_pieces, t_sign = GameStart().game_init(is_start_round=is_start_round)
        round_monitor = DominoMonitor(
            player_pieces=p_pieces,
            opponent_pieces=o_pieces,
            stock_pieces=s_pieces,
            board_pieces=b_pieces,
            turn_sign=t_sign
        )
        # 初始状态
        self.get_state_info(round_monitor)

        play_traces = {
            'P': [],
            'S_t': [],
            'A_t': [],
            'R': []
        }

        while True:
            if round_monitor.turn_sign == 1:
                valid_actions = get_validate_act(round_monitor.board_pieces, round_monitor.player_pieces,
                                                 is_start_round)
                self.Policy1.update_state(now_hand=round_monitor.player_pieces, now_board=round_monitor.board_pieces,
                                          opponent_num=len(round_monitor.opponent_pieces),
                                          stock_num=len(round_monitor.stock_pieces), validate_actions=valid_actions,
                                          is_start_round=is_start_round)
                round_act = self.Policy1.play()
            else:
                valid_actions = get_validate_act(round_monitor.board_pieces, round_monitor.opponent_pieces,
                                                 is_start_round)
                self.Policy2.update_state(now_hand=round_monitor.opponent_pieces, now_board=round_monitor.board_pieces,
                                          opponent_num=len(round_monitor.player_pieces),
                                          stock_num=len(round_monitor.stock_pieces), validate_actions=valid_actions,
                                          is_start_round=is_start_round)
                round_act = self.Policy2.play()

            if len(valid_actions) > 1:
                play_traces['S_t'].append(round_monitor.get_state())
                play_traces['P'].append(round_monitor.turn_sign)
                play_traces['A_t'].append(convert_action_to_index(round_act))

            round_win_type = round_monitor.act_state_update(round_act)
            # 打印局面信息
            self.get_state_info(round_monitor)
            # 统计模型出牌top1是否符合规则
            if 'rank' in round_act:
                self.model_action_rank.append(round_act['rank'])

            # 判断当局游戏是否结束
            if round_win_type is None:
                continue

            # 补充奖励
            for i in range(len(play_traces['P'])):
                if play_traces['P'][i] > 0:
                    # player 1 的奖励
                    play_traces['R'].append(round_win_type)
                else:
                    # player 2 的奖励
                    play_traces['R'].append(-round_win_type)
            self.print_final_info(round_win_type)
            break
        # 返回 先手次序,胜者分,出牌记录
        return t_sign, round_win_type, play_traces
