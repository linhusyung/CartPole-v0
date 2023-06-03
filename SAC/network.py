from torch import nn
from collections import deque
from random import sample
import torch.nn.functional as F
import torch
from torch.distributions import Normal


def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class Q_net(nn.Module):
    def __init__(self, num_state, num_action):
        super(Q_net, self).__init__()
        # Q1
        self.linear1_q1 = nn.Linear(num_state + num_action, 256)
        self.linear2_q1 = nn.Linear(256, 128)
        self.linear3_q1 = nn.Linear(128, 64)
        self.linear4_q1 = nn.Linear(64, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        input = torch.cat([state, action], 1)
        Q1 = F.relu(self.linear1_q1(input))
        Q1 = F.relu(self.linear2_q1(Q1))
        Q1 = F.relu(self.linear3_q1(Q1))
        Q1 = self.linear4_q1(Q1)

        return Q1


class Actor(nn.Module):
    def __init__(self, num_input, num_action):
        super(Actor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear1 = nn.Linear(num_input, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)

        self.mean = nn.Linear(32, num_action)
        self.std = nn.Linear(32, num_action)

        self.apply(weights_init_)

    def forward(self, state):
        out = F.relu(self.linear1(state))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        mean = self.mean(out)
        log_std = self.std(out)
        std = F.softplus(log_std)
        dist = Normal(mean, std)

        normal_sample = dist.rsample()  # 在标准化正态分布上采样
        log_prob = dist.log_prob(normal_sample)  # 计算该值的标准正太分布上的概率
        action = torch.tanh(normal_sample)  # 对数值进行tanh
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(2*(1 - torch.tanh(action).pow(2)) + 1e-7)  # 为了提升目标对应的概率值
        action = action * 2  # 对action求取范围
        return action, log_prob

    def train_forward(self, state):
        out = F.relu(self.linear1(state))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        mean = self.mean(out)
        log_std = self.std(out)
        std = F.softplus(log_std)
        dist = Normal(mean, std)

        normal_sample = dist.rsample()  # 在标准化正态分布上采样
        log_prob = dist.log_prob(mean)  # 计算该值的标准正太分布上的概率
        action = torch.tanh(mean)  # 对数值进行tanh
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(2*(1 - torch.tanh(action).pow(2)) + 1e-7)  # 为了提升目标对应的概率值
        action = action * 2  # 对action求取范围
        return action, log_prob




class Replay_Buffers():
    def __init__(self, batch):
        self.buffer_size = 100000
        self.buffer = deque([], maxlen=self.buffer_size)
        self.batch = batch

    def write_Buffers(self, state, next_state, reward, action, done):
        once = {'state': state, 'next_state': next_state, 'reward': reward, 'action': action, 'done': done}
        self.buffer.append(once)
        if len(self.buffer) > self.batch:
            return sample(self.buffer, self.batch)
