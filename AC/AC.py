import gym
import torch
import numpy as np
from network_AC import *
import matplotlib.pyplot as plt


class agent():
    def __init__(self):
        self.observation = env.reset()
        self.Actor = Actor(state_dim=len(self.observation))
        self.Critic = Critic(state_dim=len(self.observation))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epoch = 500
        self.eps = 0.99
        self.replay_buffers = Replay_Buffers()
        self.gamma = 0.9
        self.loss_fn = torch.nn.MSELoss()
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def np_to_tensor(self, np_data):
        return torch.tensor(np_data, dtype=torch.float32)

    def choose_action(self, out):
        if np.random.random() < self.eps:
            return np.random.randint(0, 2)
        else:
            return int(out.argmax().cpu().numpy())

    def tarin(self, replay):
        pass


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    a = agent()
    # state = a.np_to_tensor(a.observation)
    # Q = a.Critic(state).to(a.device)
    # pi = a.Actor(state).to(a.device)
    # print('Q',Q,'pi',pi)
    for i in range(a.epoch):
        observation = env.reset()
        while True:
            pass