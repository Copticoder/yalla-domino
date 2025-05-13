import random

from Policy.BasePolicy import BasePolicy
from treelib import Tree, Node

from env.DominoMonitor import DominoMonitor
from env.utils import get_validate_act, guess_op_and_stock_cards, card_list_to_str


class MCTSPolicy(BasePolicy):
    """MCTS"""

    def __init__(self, **kwargs):
        super(MCTSPolicy, self).__init__()

    def init(self, play_policy, opponent_policy):
        self.play_policy = play_policy
        self.opponent_policy = opponent_policy

    def simulate(self):
        """Simulate"""
        # 1. Create a tree root with the current state
        root_data = {
            'state': self.state.copy(),
            'valid_actions': self.validate_actions.copy(),
            'action': None,
            'visit': 0,
            'reward': 0
        }
        tree = Tree()
        tree.create_node(identifier='root', data=root_data)
        # 2. Simulate
        simu_time = 0

        while simu_time < 500:
            op_cards_guess, stock_cards_guess = guess_op_and_stock_cards(self.state['hand'], self.state['board'],
                                                                         self.state['op_num'])
            simu_monitor = DominoMonitor(
                player_pieces=self.state['hand'].copy(),
                opponent_pieces=op_cards_guess.copy(),
                stock_pieces=stock_cards_guess.copy(),
                board_pieces=self.state['board'].copy(),
                turn_sign=1
            )
            cur = 'root'
            while True:
                if simu_monitor.turn_sign == 1:
                    valid_actions = get_validate_act(simu_monitor.board_pieces, simu_monitor.player_pieces,
                                                     self.is_start_round)
                    self.play_policy.update_state(now_hand=simu_monitor.player_pieces,
                                                  now_board=simu_monitor.board_pieces,
                                                  opponent_num=len(simu_monitor.opponent_pieces),
                                                  stock_num=len(simu_monitor.stock_pieces),
                                                  validate_actions=valid_actions,
                                                  is_start_round=self.is_start_round)
                    if cur == 'root':
                        round_act = random.choice(valid_actions)
                    else:
                        round_act = self.play_policy.play()
                    if round_act not in [n.data['action'] for n in tree.children(cur)]:
                        node = Node(data={'action': round_act, 'visit': 1, 'reward': 0})
                        tree.add_node(node, parent=cur)
                        cur = node.identifier
                    else:
                        for node in tree.children(cur):
                            if node.data['action'] == round_act:
                                node.data['visit'] += 1
                                cur = node.identifier
                else:
                    valid_actions = get_validate_act(simu_monitor.board_pieces, simu_monitor.opponent_pieces,
                                                     self.is_start_round)
                    self.opponent_policy.update_state(now_hand=simu_monitor.opponent_pieces,
                                                      now_board=simu_monitor.board_pieces,
                                                      opponent_num=len(simu_monitor.player_pieces),
                                                      stock_num=len(simu_monitor.stock_pieces),
                                                      validate_actions=valid_actions,
                                                      is_start_round=self.is_start_round)
                    round_act = self.opponent_policy.play()
                simu_win_score = simu_monitor.act_state_update(round_act)
                if simu_win_score is None:
                    continue
                # Backtrack reward
                cur_node = tree.get_node(cur)
                while cur_node.identifier != 'root':
                    cur_node.data['reward'] += simu_win_score
                # print(tree)
                break
            simu_time += 1
        max_action = None
        max_reward = -999.
        for n in tree.children('root'):
            E_Reward = n.data['reward'] / (n.data['visit'] + 1e-6)
            if E_Reward >= max_reward:
                max_reward = E_Reward
                max_action = n.data['action']
        return max_action

    # Use card playing network
    def play(self, **kwargs):
        """
        Use policy network to play cards
        :param model:
        :return:
        """
        if len(self.validate_actions) > 1:
            return self.simulate()
        else:
            return self.validate_actions[0]

    def type(self):
        """Policy type"""
        return "MCTS"
