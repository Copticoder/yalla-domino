import sys
import os
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from env.utils import card_map, convert_card_to_index

"""
Dominos card playing network, learning rules.
Evaluation method, see if the top1 card played in the game is legal.
"""


def parseStateAsModelInput(state):
    """Parse the record into the network input of SLPolicyModel"""
    hand = [convert_card_to_index(c) for c in state['hand']]
    hand.sort()
    hand += [0] * (21 - len(hand))
    # 牌桌序列填充至28位长度
    board = [convert_card_to_index(c) for c in state['board']]
    board.sort()
    board += [0] * (28 - len(board))

    if len(state['board']) > 0:
        board_left, board_right = [state['board'][0][0]], [state['board'][-1][-1]]
    else:
        board_left, board_right = [7], [7]

    return np.array(hand + board + board_left + board_right + [state['op_num']] + [state['stock_num']])


class ActionAndStateValueModel(torch.nn.Module):
    """
    ActionValueModel
    """

    def __init__(self, **kwargs):
        super(ActionAndStateValueModel, self).__init__()
        self.numbers = [0, 1, 2, 3, 4, 5, 6]
        self.embedding_dim = 32
        self.inner_dim = 256
        self.cards_total_num = 28
        self.cards_on_hand_max = 21
        self.cards_stack_max = 14
        # Hand status-28 | Board status-28
        self.cards = torch.nn.Embedding(len(card_map) + 1, self.embedding_dim, padding_idx=0)
        # The points on the left/right end of the table [0, 6] correspond to the points, 7 corresponds to no card
        self.numbers = torch.nn.Embedding(len(self.numbers) + 1, self.embedding_dim)
        # Number of opponent's hand cards|[0, 21]
        self.card_num_on_vs = torch.nn.Embedding(self.cards_on_hand_max + 1, self.embedding_dim)
        # Number of cards in stock|[0,14]
        self.card_num_in_stock = torch.nn.Embedding(self.cards_stack_max + 1, self.embedding_dim)

        # Up to 21 cards in hand, up to 28 cards on the table
        self.backbone_in_dim = (21 + 28 + 1 + 1 + 1 + 1) * self.embedding_dim
        self.backbone = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.backbone_in_dim),
            torch.nn.Linear(self.backbone_in_dim, self.inner_dim),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(self.inner_dim),
            torch.nn.Linear(self.inner_dim, self.inner_dim),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(self.inner_dim),
            torch.nn.Linear(self.inner_dim, self.inner_dim),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(self.inner_dim)
        )
        self.value_head = torch.nn.Linear(self.inner_dim, 1)
        self.action_head = torch.nn.Sequential(
            torch.nn.Linear(self.inner_dim, self.cards_total_num * 2),
            torch.nn.Sigmoid()
        )

        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'

    def forward(self, data):
        """Play card"""
        cards_on_hand_features = self.cards(data[:, :21])
        cards_on_board_features = self.cards(data[:, 21:49])
        left_number_on_board_features = self.numbers(data[:, 49])
        right_number_on_board_features = self.numbers(data[:, 50])
        cards_num_on_vs_features = self.card_num_on_vs(data[:, 51])
        stock_num_features = self.card_num_in_stock(data[:, 52])

        features = torch.cat([cards_on_hand_features,
                              cards_on_board_features,
                              left_number_on_board_features.unsqueeze(1),
                              right_number_on_board_features.unsqueeze(1),
                              cards_num_on_vs_features.unsqueeze(1),
                              stock_num_features.unsqueeze(1)], dim=1)
        features = self.backbone(torch.flatten(features, 1))
        value_pred = self.value_head(features)
        action_pred = self.action_head(features)
        return action_pred, value_pred
