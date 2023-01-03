import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from network import *
from torch import optim
from torch.utils.tensorboard import SummaryWriter


class agent():
    def __init__(self, num_state, num_action, q_lr, pi_lr, target_entropy, gamma, tau, alpha_lr):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Q_net1 = Q_net(num_state, num_action).to(self.device)
        self.Q_net2 = Q_net(num_state, num_action).to(self.device)

        self.Q_net1_target = Q_net(num_state, num_action).to(self.device)
        self.Q_net2_target = Q_net(num_state, num_action).to(self.device)

        self.Q_net1_target.load_state_dict(self.Q_net1_target.state_dict())
        self.Q_net2_target.load_state_dict(self.Q_net2_target.state_dict())

        self.actor = Actor(num_state, num_action).to(self.device)

        self.batch = 64
        self.Buffers = Replay_Buffers(self.batch)
        self.gamma = 0.99

        self.q1_optimizer = optim.Adam(self.Q_net1.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.Q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.actor.parameters(), lr=pi_lr)

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau

    def np_to_tensor(self, data):
        # return torch.from_numpy(data).unsqueeze(0).float().to(self.device)
        return torch.from_numpy(data).float().to(self.device)

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def batch_resize(self, replay):
        state = torch.zeros(self.batch, 3).to(self.device)
        next_state = torch.zeros(self.batch, 3).to(self.device)
        reward = torch.zeros(self.batch, 1).to(self.device)
        action = torch.zeros(self.batch, 1).to(self.device)
        done = torch.zeros(self.batch, 1).to(self.device)

        for _ in range(len(replay)):
            state[_] = replay[_]['state']
            next_state[_] = replay[_]['next_state']
            reward[_] = replay[_]['reward']
            action[_] = self.np_to_tensor(replay[_]['action'])
            done[_] = replay[_]['done']

        return state, next_state, reward, action, done

    def train(self, replay):
        state, next_state, reward, action, done = self.batch_resize(replay)
        # 更新Q網路
        action_next, log_prob = a.actor.sample(next_state)
        entropy = -log_prob
        Q1_value = self.Q_net1_target(next_state, action_next)
        Q2_value = self.Q_net2_target(next_state, action_next)
        next_value = torch.min(Q1_value, Q2_value) + self.log_alpha.exp() * entropy
        td_target = reward + self.gamma * next_value * (1 - done)

        critic_1_loss = torch.mean(F.mse_loss(self.Q_net1(state, action), td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(self.Q_net2(state, action), td_target.detach()))

        self.q1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.q2_optimizer.step()

        # 更新actor
        new_actions, log_prob = self.actor(state)
        entropy = -log_prob
        q1_value = self.Q_net1(state, new_actions)
        q2_value = self.Q_net2(state, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        # actor_loss = ((self.log_alpha * new_actions) - torch.min(q1_value, q2_value)).mean()
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.Q_net1, self.Q_net1_target)
        self.soft_update(self.Q_net2, self.Q_net2_target)

    def tensor_to_numpy(self, data):
        return data.detach().cpu().numpy()

    def save_(self):
        torch.save(self.actor.state_dict(), 'model/actor')
        torch.save(self.Q_net1.state_dict(), 'model/Q_net1.pth')
        torch.save(self.Q_net2.state_dict(), 'model/Q_net2.pth')

    def save_best(self):
        torch.save(self.actor.state_dict(), 'model/best_actor')


if __name__ == '__main__':
    writer = SummaryWriter()

    env = gym.make('Pendulum-v0')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    pi_lr = 3e-4
    q_lr = 3e-3
    alpha_lr = 3e-4
    target_entropy = -env.action_space.shape[0]
    gamma = 0.99
    tau = 0.005
    a = agent(s_dim, a_dim, q_lr, pi_lr, target_entropy, gamma, tau, alpha_lr)
    reward_sum = []
    reward_mean = []
    b_list = []

    for _ in range(200):
        print(f'第{_}次遊戲')
        observation = env.reset()
        reward = 0
        b_list.append(_)
        while True:
            # env.render()
            state = a.np_to_tensor(observation)
            action, _ = a.actor(state)
            # print(action, state.shape)
            action = a.tensor_to_numpy(action)
            # print(action)
            observation, r, done, info = env.step(action)
            next_state = a.np_to_tensor(observation)

            replay = a.Buffers.write_Buffers(state, next_state, r, action, done)
            reward += r

            if replay is not None:
                a.train(replay)

            if done:
                print('reward', reward)
                reward_sum.append(reward)
                reward_mean.append(np.mean(reward_sum))
                writer.add_scalar("reward", reward)
                # if reward == max(reward_sum):
                #     a.save_best()
                break

    l1, = plt.plot(b_list, reward_mean)
    l2, = plt.plot(b_list, reward_sum, color='red', linewidth=1.0, linestyle='--')
    plt.legend(handles=[l1, l2], labels=['reward_mean', 'reward_sum'], loc='best')
    plt.show()
    a.save_()
