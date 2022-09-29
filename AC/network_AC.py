import torch
from torch import nn
from collections import deque
from random import sample
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim):
        super(Actor, self).__init__()
        self.Actor_h1 = nn.Linear(state_dim, 512)
        self.Actor_h1.weight.data.normal_(0, 0.1)

        self.Actor_h2 = nn.Linear(512, 256)
        self.Actor_h2.weight.data.normal_(0, 0.1)

        self.Actor_h3 = nn.Linear(256, 64)
        self.Actor_h3.weight.data.normal_(0, 0.1)

        self.Actor_h4 = nn.Linear(64, 2)
        self.Actor_h4.weight.data.normal_(0, 0.1)

        self.softmax=nn.Softmax(dim=-1)
    def forward(self, x):
        x = F.relu(self.Actor_h1(x))
        x = F.relu(self.Actor_h2(x))
        x = F.relu(self.Actor_h3(x))
        x = self.softmax(self.Actor_h4(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.Critic_h1 = nn.Linear(state_dim, 512)
        self.Critic_h1.weight.data.normal_(0, 0.1)

        self.Critic_h2 = nn.Linear(512, 256)
        self.Critic_h2.weight.data.normal_(0, 0.1)

        self.Critic_h3 = nn.Linear(256, 64)
        self.Critic_h3.weight.data.normal_(0, 0.1)

        self.Critic_h4 = nn.Linear(64, 2)
        self.Critic_h4.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.Critic_h1(x))
        x = F.relu(self.Critic_h2(x))
        x = F.relu(self.Critic_h3(x))
        x = self.Critic_h4(x)
        return x


# 經驗回放
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
