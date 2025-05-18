import random
from Policy.BasePolicy import BasePolicy


class RandomPolicy(BasePolicy):
    def __init__(self, **kwargs):
        super(RandomPolicy, self).__init__()

    # 随机从动作空间中采样一个动作
    def play(self, **kwargs):
        sample_act = random.sample(self.validate_actions, k=1)[0]
        return sample_act

    def type(self):
        """策略类型"""
        return "Random"
