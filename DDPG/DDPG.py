# 角速度，角速度，水平速度，垂直速度，关节位置和关节角速度，腿是否与地面的接触以及10个激光雷达测距仪的测量值
# 獎勵前進，總共 300+ 點到遠端。如果機器人摔倒，
# 它得到-100。應用電機扭矩花費少量積分，更優代理
# 會得到更好的分數。
# 要解決這個遊戲，你需要在 1600 個時間步中獲得 300 分。
# 要解決硬核版本，您需要 2000 個時間步長中的 300 個點。
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from DDPG_network import *

np.random.seed(0)
class agent():
    def __init__(self):
        self.epoch = 400
        self.state_dim = len(env.reset())

        self.Actor = Actor(self.state_dim)
        self.target_Actor = Actor(self.state_dim)
        self.target_Actor.load_state_dict(self.target_Actor.state_dict())

        self.Critic = Critic(self.state_dim, 1)
        self.target_Critic = Critic(self.state_dim, 1)
        self.target_Critic.load_state_dict(self.Critic.state_dict())

        self.loss = torch.nn.MSELoss()
        self.replay = Replay_Buffers()
        self.gamma = 0.99
        self.Critic_optimizer = torch.optim.Adam(self.Critic.parameters(), lr=0.001)
        self.Actor_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=0.002)
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
            action).cpu(), self.to_tensor(done).unsqueeze(1)

    def soft_update(self, target, source, t):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)

    def train(self, replay):
        state, state_next, reward, action, done = self.read_replay(replay)

        # updata Critic
        with torch.no_grad():
            target_action = self.target_Actor(state_next).to(self.device)
            target_Q = self.target_Critic(state_next, target_action.cpu()).to(self.device)
            target = reward.to(self.device) + self.gamma * target_Q * (1 - done).to(self.device)
        Q = self.Critic(state, action.detach()).to(self.device)
        loss = self.loss(target, Q)
        self.Critic_optimizer.zero_grad()
        loss.backward()
        self.Critic_optimizer.step()

        # updata Actor
        mu = self.Actor(state).to(self.device)
        u_A_q = self.Critic(state, mu.cpu()).to(self.device)
        actor_loss = -torch.mean(u_A_q)
        self.Actor_optimizer.zero_grad()
        actor_loss.backward()
        self.Actor_optimizer.step()

        # updata Actor_target
        self.soft_update(self.target_Actor, self.Actor, 0.005)
        # updata Critic_target
        self.soft_update(self.target_Critic, self.Critic, 0.005)

    def save_model(self, path: str):
        torch.save(self.Actor.state_dict(), path)


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    a = agent()
    reward_list = []
    i_list = []
    for i in range(a.epoch):
        observation = env.reset()
        reward_sum = 0
        i_list.append(i)
        print('第', i, '次遊戲')
        while True:
            # env.render()
            state_now = a.np_to_tensor(observation)
            action = a.choose_action(a.Actor(state_now).to(a.device))

            observation, reward, done, info = env.step(a.tensor_to_np(action))
            replay_sample = a.replay.write_Buffers(state_now, a.np_to_tensor(observation), reward, action, done)
            if replay_sample is not None:
                a.train(replay_sample)

            reward_sum += reward
            if done:
                reward_list.append(reward_sum)
                print(reward_sum,'max',max(reward_list))
                if reward_sum == max(reward_list):
                    a.save_model('model/model_params_max.pth')
                break
    a.save_model('model/model_params.pth')
    print(max(reward_list))
    plt.plot(i_list, reward_list)
    plt.show()
