import os
import argparse
import os.path
from torch.cuda.amp import GradScaler
import torch
from torch.utils.data import dataset
from torch.utils.data import DataLoader
from torch import optim
import tqdm
import numpy as np
import random
from prefetch_generator import BackgroundGenerator

from env.utils import card_map
from env.PlayGame import PlayGame
from Policy import init_policy
from utils import parse_config
from models.SLModel import parseStateAsModelInput, SLModel

"""
dominos出牌网络，学习规则。
评估方式，看对局中top1的出牌，是否合法。
"""


class RewardMSELoss(torch.nn.Module):
    """自监督对比损失"""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, preds, targets, rewards):
        """计算loss"""
        pred_rewards = torch.gather(preds, 1, targets)
        return torch.nn.functional.mse_loss(pred_rewards, rewards)


class ValueActionLoss(torch.nn.Module):
    """loss for Value network && Action policy network"""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, pred_values, pred_actions, targets, rewards):
        """计算loss"""
        value_loss = torch.nn.functional.mse_loss(pred_values, rewards)
        action_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_actions, targets)
        return value_loss + action_loss


class DataLoaderX(DataLoader):
    """
    speed up
    """

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def collate_function(data):
    """处理批量数据"""
    states_np = np.zeros([len(data), data[0][0].shape[0]], dtype=np.int64)
    targets_np = np.zeros([len(data), 1], dtype=np.int64)
    rewards_np = np.zeros([len(data), 1], dtype=np.float32)
    for i, (state, target, reward) in enumerate(data):
        states_np[i, :] = state
        targets_np[i, :] = target
        rewards_np[i, :] = reward

    return torch.from_numpy(states_np), torch.from_numpy(targets_np), torch.from_numpy(rewards_np)


class DominoDataset(dataset.Dataset):
    """
    模拟随机对局产生State,Action
    """

    def __init__(self, player1_configs, player2_configs, data_max_gen=10000000, sample_policy='sample_random'):
        "加载record_file,每行为一个s_t,a_t,r_t"
        self.data_max_gen = data_max_gen
        self.sample_policy = sample_policy
        p1 = init_policy(player1_configs)
        p2 = init_policy(player2_configs)
        self.g = PlayGame(p1, p2)

    def __len__(self):
        return self.data_max_gen

    def sample_random(self, ):
        """随机采集一条记录,返回S_t,A_t,R"""
        """随机采集一条胜者的记录,返回S_t,A_t,R"""
        while 1:
            t_sign, round_win_type, play_traces = self.g.run_game()
            if round_win_type == 0:
                continue
            if len(play_traces['P']) == 0:
                continue
            # 随机返回一次(S_t,A_t)
            choice = random.choice(range(len(play_traces['P'])))
            state = play_traces['S_t'][choice]
            target = play_traces['A_t'][choice]
            R = play_traces['R'][choice]
            return parseStateAsModelInput(state), target - 1, R

    def __getitem__(self, index):
        """将记录解析为训练数据"""
        return self.sample_random()


def train(dataLoader, epoch_num=10, save_path="./weights/", pretrain=None, device='cpu'):
    """训练"""
    # 加载模型
    model = SLModel(device=device)
    if pretrain is not None:
        model.load_state_dict(torch.load(pretrain))
    if model.device == 'cuda':
        model.cuda()

    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    # 定义学习率策略
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=1e-6)
    # 定义损失函数
    loss_func = RewardMSELoss()
    gradScaler = GradScaler(enabled=model.device == 'cuda')
    # 训练epochs
    for epoch in range(epoch_num):
        print("start epoch {}".format(epoch))
        loop = tqdm.tqdm(dataLoader, total=len(dataLoader))
        for states, targets, rewards in loop:
            optimizer.zero_grad()
            with torch.autocast(device_type=model.device, enabled=model.device == 'cuda'):
                if model.device == 'cuda':
                    states = states.cuda()
                    targets = targets.cuda()
                    rewards = rewards.cuda()
                preds = model.forward(states)
                loss = loss_func(preds, targets, rewards)
                if model.device == 'cuda':
                    gradScaler.scale(loss).backward()
                    gradScaler.step(optimizer)
                    gradScaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            acc = torch.sum(torch.argmax(preds, dim=1) == targets.squeeze()) / targets.shape[0]
            loop.set_postfix(epoch=epoch, loss="{:.6f}".format(loss.item()),
                             acc="{:.6f}".format(acc),
                             lr=optimizer.param_groups[0]['lr'])
        lr_scheduler.step()
        # 保存模型
        torch.save(model.state_dict(), os.path.join(save_path, "params_epoch{}.pth".format(epoch)))


def parseParams():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-save_path', type=str, help='model save path.', default=None)
    parser.add_argument('-epoch', type=int, help='train epoch num', default=10)
    parser.add_argument('-task', type=str, help='train | eval | prepare_data', default='train')
    parser.add_argument('-batch_size', type=int, help='train epoch num', default=1024)
    parser.add_argument('-train_game_num', type=int, help='train game num', default=10000000)
    parser.add_argument('-pretrain', type=str, help='pretrain model path', default=None)
    parser.add_argument('-policy1_config', type=str, help='策略配置文件', default=None)
    parser.add_argument('-policy2_config', type=str, help='策略配置文件', default=None)
    parser.add_argument('-sample_policy', type=str, help='样本采样方式', default='sample_random')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    params = parseParams()
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn')
    if params.task == 'train':
        if not os.path.exists(params.save_path):
            os.makedirs(params.save_path)
            # 加载数据集
        domino_dataset = DominoDataset(data_max_gen=params.train_game_num,
                                       player1_configs=parse_config(params.policy1_config),
                                       player2_configs=parse_config(params.policy2_config),
                                       sample_policy=params.sample_policy)
        dataloader = DataLoaderX(domino_dataset, batch_size=params.batch_size, num_workers=16, pin_memory=True,
                                 shuffle=True, drop_last=True, collate_fn=collate_function)

        train(dataloader, save_path=params.save_path, epoch_num=params.epoch, pretrain=params.pretrain)
