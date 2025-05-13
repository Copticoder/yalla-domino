import random
from Policy.BasePolicy import BasePolicy


class RandomPolicy(BasePolicy):
    def __init__(self, **kwargs):
        super(RandomPolicy, self).__init__()

    # Randomly sample an action from the action space
    def play(self, **kwargs):
        sample_act = random.sample(self.validate_actions, k=1)[0]
        return sample_act

    def type(self):
        """Policy type"""
        return "Random"
