import random
from env.GameStart import GameStart
from env.DominoMonitor import DominoMonitor
from env.utils import get_validate_act, convert_action_to_index


class PlayGame:
    """Play game"""

    def __init__(self, Policy1, Policy2, print_details=False, **kwargs):
        # Print game information
        self.print_details = print_details

        self.Policy1 = Policy1
        self.Policy2 = Policy2

        self.model_action_rank = []

    def get_state_info(self, game_state):
        """
        Print game information
        :param game_state:
        """
        if not self.print_details:
            return
        print("-" * 50)
        print("Player1 Hand:",
              len(game_state.player_pieces) if self.Policy2.type() == 'human' else game_state.player_pieces)
        print("Player2 Hand:",
              len(game_state.opponent_pieces) if self.Policy1.type() == 'human' else game_state.opponent_pieces)
        print("Board:", game_state.board_pieces)
        if self.Policy1.type() != 'human' and self.Policy2.type() != 'human':
            print("Stock:", game_state.stock_pieces)
        else:
            print("Stock:", len(game_state.stock_pieces))
        print("-" * 50)

    def get_init_info(self):
        """
        Print game initialization information
        :param game_state:
        """
        if not self.print_details:
            return
        print("#" * 25 + "Game Start" + "#" * 25)

    def print_final_info(self, round_win_score):
        """
        Print game end information
        :param:
        """
        if not self.print_details:
            return
        if round_win_score > 0:
            print("This round is over, Player1 wins~, score", round_win_score)
        elif round_win_score < 0:
            print("This round is over, Player2 wins~, score", round_win_score)
        else:
            print("Draw.")

        print("#" * 25 + "Game Ended" + "#" * 25)

    def run_game(self):
        """Play a game"""
        self.get_init_info()
        # Generate round 1 game with 1/4 probability
        is_start_round = random.random() > 0.75
        p_pieces, o_pieces, s_pieces, b_pieces, t_sign = GameStart().game_init(is_start_round=is_start_round)
        round_monitor = DominoMonitor(
            player_pieces=p_pieces,
            opponent_pieces=o_pieces,
            stock_pieces=s_pieces,
            board_pieces=b_pieces,
            turn_sign=t_sign
        )
        # Initial state
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
            # Print board information
            self.get_state_info(round_monitor)
            # Count whether the top1 card played by the model conforms to the rules
            if 'rank' in round_act:
                self.model_action_rank.append(round_act['rank'])

            # Judge whether the current game is over
            if round_win_type is None:
                continue

            # Supplement rewards
            for i in range(len(play_traces['P'])):
                if play_traces['P'][i] > 0:
                    # Player 1's reward
                    play_traces['R'].append(round_win_type)
                else:
                    # Player 2's reward
                    play_traces['R'].append(-round_win_type)
            self.print_final_info(round_win_type)
            break
        # Return starting order, winner's score, play record
        return t_sign, round_win_type, play_traces
