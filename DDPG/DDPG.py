# 角速度，角速度，水平速度，垂直速度，关节位置和关节角速度，腿是否与地面的接触以及10个激光雷达测距仪的测量值
# 獎勵前進，總共 300+ 點到遠端。如果機器人摔倒，
# 它得到-100。應用電機扭矩花費少量積分，更優代理
# 會得到更好的分數。
# 要解決這個遊戲，你需要在 1600 個時間步中獲得 300 分。
# 要解決硬核版本，您需要 2000 個時間步長中的 300 個點。
import gym
import torch
import numpy as np

from DDPG_network import *


class agent():
    def __init__(self):
        self.epoch = 10
        self.state_dim = len(env.reset())

        self.Actor = Actor(self.state_dim)
        self.target_Actor = Actor(self.state_dim)

        self.Critic = Critic(self.state_dim, 4)
        self.target_Critic = Critic(self.state_dim, 4)

        self.loss = torch.nn.MSELoss()
        self.replay = Replay_Buffers()
        self.gamma = 0.9
        self.optimizer = torch.optim.Adam(self.Critic.parameters(), lr=1e-4)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float32)

    def np_to_tensor(self, data):
        return torch.from_numpy(data).to(torch.float32)

    def tuple_of_tensor_to_tensor(self, tuplt_of_tensor):
        return torch.stack(tuplt_of_tensor, dim=0)

    def tensor_to_np(self, data):
        return data.cpu().detach().numpy()

    def choose_action(self, action):
        noise = torch.tensor(np.random.normal(loc=0.0, scale=0.1), dtype=torch.float32).to(self.device)
        return torch.clamp(action + noise, -1, 1)

    def read_replay(self, replay):
        state = []
        state_next = []
        reward = []
        action = []
        done = []
        for i in range(len(replay)):
            state.append(replay[i]['state'])
            state_next.append(replay[i]['next_state'])
            reward.append(replay[i]['reward'])
            action.append(replay[i]['action'])
            done.append(replay[i]['done'])
        return self.tuple_of_tensor_to_tensor(state), self.tuple_of_tensor_to_tensor(
            state_next), self.to_tensor(reward).unsqueeze(1), self.tuple_of_tensor_to_tensor(
            action).cpu(), done

    def train(self, replay):
        state, state_next, reward, action, done = self.read_replay(replay)

        # updata Critic
        target_action = self.target_Actor(state_next).to(self.device).detach()
        target_Q = self.target_Critic(state_next, target_action.cpu()).to(self.device).detach()
        target = reward.to(self.device) + self.gamma * target_Q

        Q = self.Critic(state, action).to(self.device)
        loss = self.loss(target, Q)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        # updata Actor
        

if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3')
    a = agent()

    for i in range(a.epoch):
        observation = env.reset()
        print('第', i, '次遊戲')
        while True:
            # env.render()
            state_now = a.np_to_tensor(observation)
            action = a.choose_action(a.Actor(state_now).to(a.device))

            observation, reward, done, info = env.step(a.tensor_to_np(action))

            replay_sample = a.replay.write_Buffers(state_now, a.np_to_tensor(observation), reward, action, done)
            if replay_sample is not None:
                a.train(replay_sample)
                print('asdfasf')
            if done:
                break
