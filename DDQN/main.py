import gym
import torch
import numpy as np
from network_torch import DQN_torch, Replay_Buffers
import matplotlib.pyplot as plt
import copy


class agent():
    def __init__(self):
        self.observation = env.reset()
        self.model = DQN_torch(state_dim=len(self.observation))
        self.target_model=DQN_torch(state_dim=len(self.observation))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epoch = 500
        self.eps = 0.99
        self.replay_buffers = Replay_Buffers()
        self.gamma = 0.9
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def np_to_tensor(self, np_data):
        return torch.tensor(np_data, dtype=torch.float32)

    def choose_action(self, out):
        if np.random.random() < self.eps:
            return np.random.randint(0, 2)
        else:
            return int(out.argmax().cpu().numpy())

    def tarin(self, replay):
        state_ = []
        next_state_ = []
        action_ = []
        reward_ = []
        done_ = []
        for i in range(len(replay)):
            state_.append(replay[i]['state'])
            next_state_.append(replay[i]['next_state'])
            action_.append(replay[i]['action'])
            reward_.append(replay[i]['reward'])
            done_.append(replay[i]['done'])
        state_, next_state_, action_, reward_, done_ = \
            tuple(state_), tuple(next_state_), self.np_to_tensor(action_).to(self.device), self.np_to_tensor(
                reward_).to(self.device), self.np_to_tensor(done).to(self.device)

        Q=self.model(self.np_to_tensor(state_)).to(self.device)
        #狀態t，給dqn
        Q_next = self.model(self.np_to_tensor(next_state_)).to(self.device)
        argmaz_action = torch.argmax(Q_next, 1)
        #下一個狀態給dqn選最大的
        Q_target_net = self.target_model(self.np_to_tensor(next_state_)).to(self.device)
        max_Q_ = Q_target_net.gather(1, argmaz_action.unsqueeze(1).type(torch.int64)).squeeze(1)
        #下一個狀態給Q_target,重dqn選最大的index給Q‘
        Q_target = reward_ + self.gamma * max_Q_ * (1 - done_)
        Q_eval = Q.gather(1, action_.unsqueeze(1).type(torch.int64)).squeeze(1)

        loss = self.loss_fn(Q_target, Q_eval)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def save_model(self):
        torch.save(self.model.state_dict(), 'model/model_params.pth')

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    a = agent()
    reward_list = []
    b_list = []
    for b in range(a.epoch):
        b_list.append(b)
        reward_sum = 0
        observation = env.reset()
        print('第', b, '次遊戲')
        count=0
        while True:
            # env.render()
            if b == a.epoch - 1:
                env.render()
            state = observation
            obe = a.np_to_tensor(observation)
            out = a.model(obe).to(a.device)
            # print(out)
            action = a.choose_action(out)
            # print(action)
            observation,r, done, info = env.step(action)

            next_state = observation
            x, v, theta, omega = next_state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2
            replay = a.replay_buffers.write_Buffers(state, next_state, reward, action, done)

            reward_sum += r
            if replay is not None:
                a.tarin(replay)
            if count%5==0:
                a.target_model.load_state_dict(a.model.state_dict())
            count+=1

            if done:
                if a.eps > 0.1:
                    # a.eps -= a.eps*0.009
                    a.eps *=0.9
                print('eps', a.eps)
                reward_list.append(reward_sum)
                print(reward_sum)
                break
    a.save_model()
    plt.plot(b_list, reward_list)
    plt.show()
