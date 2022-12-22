from torch import nn
from collections import deque
from random import sample
import torch.nn.functional as F
import torch


def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class V_net(nn.Module):
    def __init__(self, num_input):
        super(V_net, self).__init__()
        self.linear1 = nn.Linear(num_input, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 1)

        self.apply(weights_init_)

    def forward(self, input):
        out = F.relu(self.linear1(input))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        out = self.linear4(out)
        return out


class Q_net(nn.Module):
    def __init__(self, num_input):
        super(Q_net, self).__init__()
        self.linear1 = nn.Linear(num_input, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 1)

        self.apply(weights_init_)

    def forward(self, input):
        out = F.relu(self.linear1(input))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        out = self.linear4(out)
        return out


class Actor(nn.Module):
    def __init__(self, num_input, num_action):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(num_input, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)

        self.linear4 = nn.Linear(32, num_action)
        self.linear4 = nn.Linear(32, num_action)

    def forward(self):
        pass


class Replay_Buffers():
    def __init__(self):
        self.buffer_size = 5000
        self.buffer = deque([], maxlen=self.buffer_size)
        self.batch = 20

    def write_Buffers(self, state, next_state, reward, action, done, action_probability):
        once = {'state': state, 'next_state': next_state, 'reward': reward, 'action': action, 'done': done,
                'action_probability': action_probability}
        self.buffer.append(once)
        if len(self.buffer) > self.batch:
            return sample(self.buffer, self.batch)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # V=V_net(10).to(device)
    V = V_net(10)
    for _ in range(100):
        a = torch.rand(10)
        # a = torch.rand(10).to(device)
        # out = V(a)
        # print(out)
        print('a', a)
        print('clamp', torch.clamp(a, min=-0.5, max=0.5))
