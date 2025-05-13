from env.utils import convert_index_to_action
import torch
import numpy as np
from models.SLModel import SLModel, parseStateAsModelInput
from Policy.BasePolicy import BasePolicy


class SLModelPolicy(BasePolicy):
    def __init__(self, **kwargs):
        super(SLModelPolicy, self).__init__()
        self.model = SLModel(**kwargs)
        self.model.load_state_dict(torch.load(kwargs['weights'], map_location='cpu'))
        self.model.eval()

    def model_play(self, states, valid_actions):
        data = parseStateAsModelInput(states)
        board_left, board_right = data[49], data[50]
        data_tensor = torch.from_numpy(data).unsqueeze(0)
        if self.model.device == 'cuda':
            data_tensor = data_tensor.cuda()
        rewards = self.model(data_tensor).detach().cpu().numpy()
        cards_idx = (np.argsort(rewards) + 1).tolist()[0][::-1]
        for rank, card_id in enumerate(cards_idx):
            action = convert_index_to_action(card_id)
            if len(self.state['board']) == 0:
                action['inverse'] = 3
            else:
                if action['direction'] == 3:
                    if action['card'][-1] == board_left:
                        action['inverse'] = 3
                    else:
                        action['inverse'] = 4
                else:
                    if action['card'][0] == board_right:
                        action['inverse'] = 3
                    else:
                        action['inverse'] = 4
            if action in valid_actions:
                round_action = action
                round_action['rank'] = rank
                return round_action

    # 使用出牌网络
    def play(self, **kwargs):
        """
        使用策略网络出牌
        :param model:
        :return:
        """
        if len(self.validate_actions) > 1:
            return self.model_play(self.state, self.validate_actions)
        return self.validate_actions[0]

    def type(self):
        """策略类型"""
        return "Model"
