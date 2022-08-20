from torch import nn
import torch.nn.functional as F
from collections import deque
from random import sample
import torch


class DQN_torch(nn.Module):
    def __init__(self, state_dim):
        super(DQN_torch, self).__init__()
        self.h1 = nn.Linear(state_dim, 512)
        self.h1.weight.data.normal_(0, 0.1)
        self.h2 = nn.Linear(512, 256)
        self.h2.weight.data.normal_(0, 0.1)
        self.h3 = nn.Linear(256, 64)
        self.h3.weight.data.normal_(0, 0.1)
        self.h4 = nn.Linear(64, 2)
        self.h4.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        x = self.h4(x)
        return x


class Dueling_network(nn.Module):
    def __init__(self, state_dim):
        super(Dueling_network, self).__init__()
        self.h1 = nn.Linear(state_dim, 512)
        self.h1.weight.data.normal_(0, 0.1)
        self.h2 = nn.Linear(512, 256)
        self.h2.weight.data.normal_(0, 0.1)
        self.h3 = nn.Linear(256, 64)
        self.h3.weight.data.normal_(0, 0.1)
        # self.h4 = nn.Linear(64, 2)
        # self.h4.weight.data.normal_(0, 0.1)
        # 共享層
        self.A1 = nn.Linear(64, 1)
        self.A1.weight.data.normal_(0, 0.1)
        # 優勢
        self.V1 = nn.Linear(64, 2)
        self.V1.weight.data.normal_(0, 0.1)
        # 狀態價值函數

    def forward(self, x):
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        a = self.A1(x)
        v = self.V1(x)
        x = v + (torch.mean(a) - a)
        return x


class Replay_Buffers():
    def __init__(self):
        self.buffer_size = 5000
        self.buffer = deque([], maxlen=self.buffer_size)
        self.batch = 20

    def write_Buffers(self, state, next_state, reward, action, done):
        once = {'state': state, 'next_state': next_state, 'reward': reward, 'action': action, 'done': done, }
        self.buffer.append(once)
        if len(self.buffer) > self.batch:
            return sample(self.buffer, self.batch)
