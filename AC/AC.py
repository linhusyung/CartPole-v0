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
        self.gamma = 0.99
        self.loss_fn = torch.nn.MSELoss()
        self.Critic_optimizer = torch.optim.Adam(self.Critic.parameters(), lr=2e-4)
        self.Actor_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=4e-4)
        self.action_space = np.array([0, 1])

    def np_to_tensor(self, np_data):
        return torch.tensor(np_data, dtype=torch.float32)

    def choose_action(self, action_probability):
        action_p = action_probability.detach().cpu().numpy()
        action = np.random.choice(self.action_space, p=action_p)
        return action

    def train(self, state, next_state, reward, action_probability, action):
        V = self.Critic(state).to(a.device)
        next_V = self.Critic(next_state).to(a.device)
        TD_error = self.np_to_tensor(reward) + self.gamma * next_V - V
        Critic_loss = self.loss_fn(self.np_to_tensor(reward) + self.gamma * next_V, V)

        self.Critic_optimizer.zero_grad()
        Critic_loss.backward()
        self.Critic_optimizer.step()

        pi = -torch.log(action_probability[action]) * TD_error.detach()
        self.Actor_optimizer.zero_grad()
        pi.backward()
        self.Actor_optimizer.step()


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    a = agent()
    reward_list = []
    i_list = []
    best_reward=0
    for i in range(a.epoch):
        i_list.append(i)
        reward_sum = 0
        observation = env.reset()
        print('第', i, '次遊戲')
        while True:
            # env.render()
            state = observation
            obe = a.np_to_tensor(observation)

            action_probability = a.Actor(obe).to(a.device)
            action = a.choose_action(action_probability)

            observation, r, done, info = env.step(action)

            next_state = observation
            reward_sum += r

            next_obe = a.np_to_tensor(next_state)

            x, v, theta, omega = next_state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2

            if best_reward==200:
                pass
            else:
                a.train(obe, next_obe, reward, action_probability, action)

            if done:
                reward_list.append(reward_sum)
                print(reward_sum)
                best_reward=reward_sum
                break
    plt.plot(i_list, reward_list)
    plt.show()
