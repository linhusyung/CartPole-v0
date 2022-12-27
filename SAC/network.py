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
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + epsilon)
        action = action * 2
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
