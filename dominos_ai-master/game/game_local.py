import sys
import os

import argparse

import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from Policy import init_policy
from utils import parse_config
from env.PlayGame import PlayGame


def parseParams():
    """Parse command line parameters"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-save_record', type=str, help='save game records or not', default=None)
    parser.add_argument('-game_num', type=int, help='game times', default=100)
    parser.add_argument('-policy1_config', type=str, help='Policy configuration file', default=None)
    parser.add_argument('-policy2_config', type=str, help='Policy configuration file', default=None)
    parser.add_argument('-print_details', action='store_true', help='print vs process or not')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    params = parseParams()
    config1 = parse_config(params.policy1_config)
    config2 = parse_config(params.policy2_config)
    p1 = init_policy(config1)
    p2 = init_policy(config2)

    g = PlayGame(p1, p2, print_details=params.print_details)
    first_move, game_result, model_action_rank = [], [], []
    for _ in tqdm.tqdm(range(params.game_num)):
        t_sign, round_win_type, play_traces = g.run_game()
        model_action_rank += g.model_action_rank
        first_move.append(t_sign)
        game_result.append(round_win_type)

    print("Player1 first move probability:", sum([m > 0 for m in first_move]) / params.game_num)
    print("Player1 win probability:", sum([g > 0 for g in game_result if g != 0]) / sum([1 for g in game_result if g != 0]))
    print("Player1 average winning score:", sum([g for g in game_result if g > 0]) / sum([1 for g in game_result if g != 0]))
    print("Player1 average losing score:", sum([g for g in game_result if g < 0]) / sum([1 for g in game_result if g != 0]))
    if len(model_action_rank) > 0:
        print("Model card playing rationality (the lower the better):", sum(model_action_rank) / len(model_action_rank))
