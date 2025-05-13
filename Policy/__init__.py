from Policy.SLModelPolicy import SLModelPolicy
from Policy.RandomPolicy import RandomPolicy
from Policy.RulePolicy import RulePolicy
from Policy.MCTSPolicy import MCTSPolicy
from Policy.HumanPolicy import HumanPolicy

register_policy = {
    "SLModel": SLModelPolicy,
    "random": RandomPolicy,
    "rule": RulePolicy,
    "human": HumanPolicy,
    "MCTS": MCTSPolicy
}


def get_policy(policy):
    """返回policy"""
    return register_policy[policy]


def init_policy(configs):
    # 初始化策略
    p = get_policy(configs['policy'])(**configs)
    if p.type() == 'MCTS':
        p.init(get_policy(configs['MCTS_policy']['policy'])(**configs['MCTS_policy']),
               get_policy(configs['simu_policy']['policy'])(**configs['simu_policy']))
    return p
