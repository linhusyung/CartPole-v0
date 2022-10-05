import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
from random import sample

class Actor(nn.Module):
    def __init__(self, state_dim):
        super(Actor, self).__init__()
        self.actor_fc1 = nn.Linear(state_dim, 64)
        self.actor_fc1.weight.data.normal_(0, 0.1)
        self.actor_fc1_ = nn.LayerNorm(64)

        self.actor_fc2 = nn.Linear(64, 32)
        self.actor_fc2.weight.data.normal_(0, 0.1)
        self.actor_fc2_ = nn.LayerNorm(32)

        self.actor_fc3 = nn.Linear(32, 1)
        self.actor_fc3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.actor_fc1(x))
        x = self.actor_fc1_(x)

        x = F.relu(self.actor_fc2(x))
        x = self.actor_fc2_(x)

        x = torch.tanh(self.actor_fc3(x))
        return x*2


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.Critic_h1 = nn.Linear(state_dim, 64)
        self.Critic_h1.weight.data.normal_(0, 0.1)
        self.Critic_h1_ = nn.LayerNorm(64)

        self.Critic_action = nn.Linear(action_dim, 32)
        self.Critic_action.weight.data.normal_(0, 0.1)

        self.Critic_g1 = nn.Linear(64 + 32, 32)
        self.Critic_g1.weight.data.normal_(0, 0.1)
        self.Critic_g1_ = nn.LayerNorm(32)

        self.Critic_g2 = nn.Linear(32, 1)
        self.Critic_g2.weight.data.normal_(0, 0.1)

    def forward(self, state, action):
        x = self.Critic_h1_(F.relu(self.Critic_h1(state)))
        action = F.relu(self.Critic_action(action))
        goal = torch.cat([x, action], dim=-1)
        x = self.Critic_g1_(F.relu(self.Critic_g1(goal)))
        x = self.Critic_g2(x)
        return x

class Replay_Buffers():
    def __init__(self):
        self.buffer_size = 1000000
        self.buffer = deque([], maxlen=self.buffer_size)
        self.batch = 1314

    def write_Buffers(self, state, next_state, reward, action, done):
        once = {'state': state, 'next_state': next_state, 'reward': reward, 'action': action, 'done': done, }
        self.buffer.append(once)
        if len(self.buffer) >self.batch:
            return sample(self.buffer, self.batch)