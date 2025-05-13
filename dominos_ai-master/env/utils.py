import random
from collections import OrderedDict

card_map = {
    '0-0': 1,
    '0-1': 8,
    '0-2': 9,
    '0-3': 10,
    '0-4': 11,
    '0-5': 12,
    '0-6': 13,
    '1-1': 2,
    '1-2': 14,
    '1-3': 15,
    '1-4': 16,
    '1-5': 17,
    '1-6': 18,
    '2-2': 3,
    '2-3': 19,
    '2-4': 20,
    '2-5': 21,
    '2-6': 22,
    '3-3': 4,
    '3-4': 23,
    '3-5': 24,
    '3-6': 25,
    '4-4': 5,
    '4-5': 26,
    '4-6': 27,
    '5-5': 6,
    '5-6': 28,
    '6-6': 7
}
# In python3.6+, keys are ordered, OrderedDict is not necessary
card_map = OrderedDict(card_map)

card_map_inv = {card_map[k]: k for k in card_map}


def card_str_to_list(card_str):
    """Card of type list"""
    return [int(card_str[0]), int(card_str[-1])]


def card_list_to_str(card_list):
    return str(min(card_list)) + '-' + str(max(card_list))


def convert_card_to_index(card):
    """Convert card to index"""
    return card_map[card_list_to_str(card)]


def convert_index_to_card(index):
    """Convert index to card"""
    card = card_map_inv[index]
    return card_str_to_list(card)


def convert_action_to_index(action):
    """Convert card playing action to index"""
    card_idx = convert_card_to_index(action['card'])
    if action['direction'] == 4:
        card_idx += len(card_map)
    return card_idx


def convert_index_to_action(index):
    """Convert index to card playing action"""
    return {'direction': 3 if index <= len(card_map) else 4,
            'card': convert_index_to_card(index if index <= len(card_map) else index - len(card_map))}


# Action legality judgment: return 3, 4 for legal, return 0 for illegal
def domino_critics(now_board, play_card, play_direction):
    if not now_board:
        return 3
    # Left side of board
    if play_direction == 3:
        # Can it match if the card is flipped?
        if now_board[0][0] == play_card[1]:
            return 3
        elif now_board[0][0] == play_card[0]:
            return 4
        else:
            return 0
    # Right side of board
    elif play_direction == 4:
        # Can it match if the card is flipped?
        if now_board[-1][-1] == play_card[0]:
            return 3
        elif now_board[-1][-1] == play_card[1]:
            return 4
        else:
            return 0
    else:
        raise ValueError("Undefined direction")


# Get legal action space: [[card, whether to flip, add to left/right side]]
# Whether to flip: 3 for no flip, 4 for flip
# Add to left/right side: 3 for left, 4 for right
def get_validate_act(now_board, now_hand, is_start_round=False):
    actions = []
    if not now_board:
        # First round, first card played must be the largest double
        if is_start_round:
            card = max([[x, y] for x, y in now_hand if x == y])
            # Largest double, default flip, default side
            actions.append(
                {
                    "card": card,
                    "direction": 3,
                    "inverse": 3
                }
            )
        else:
            for card in now_hand:
                actions.append(
                    {
                        "card": card,
                        "direction": 3,
                        "inverse": 3
                    }
                )
    else:
        board_head, board_tail = now_board[0][0], now_board[-1][-1]
        for card in now_hand:
            if card[1] == board_head:
                actions.append(
                    {
                        "card": card,
                        "direction": 3,
                        "inverse": 3
                    }
                )
            # 同点牌，同侧，动作空间不必重复添加
            if card[0] == board_head and card[0] != card[1]:
                actions.append(
                    {
                        "card": card,
                        "direction": 3,
                        "inverse": 4
                    }
                )
            # board左、右点相同
            if board_head == board_tail:
                continue

            if card[0] == board_tail:
                actions.append(
                    {
                        "card": card,
                        "direction": 4,
                        "inverse": 3
                    }
                )
            # 同点牌，同侧，动作空间不必重复添加
            if card[1] == board_tail and card[1] != card[0]:
                actions.append(
                    {
                        "card": card,
                        "direction": 4,
                        "inverse": 4
                    }
                )

    return actions


def guess_op_and_stock_cards(hand, board, op_num):
    """Guess opponent and stock hand cards"""
    board_sorted = [[min(b), max(b)] for b in board]
    left_cards = [card_str_to_list(c) for c in card_map if
                  card_str_to_list(c) not in hand and card_str_to_list(c) not in board_sorted]
    random.shuffle(left_cards)
    return left_cards[:op_num], left_cards[op_num:]
