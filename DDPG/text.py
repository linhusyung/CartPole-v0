import gym
import torch
from DDPG_network import *

class agent():
    def __init__(self):
        self.observation = env.reset()
        self.state_dim = len(env.reset())
        self.Actor = Actor(self.state_dim)
        self.epoch=20
        self.PATH='model/model_params_max.pth'
        self.Actor.load_state_dict(torch.load(self.PATH))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def tensor_to_np(self, data):
        return data.cpu().detach().numpy()

    def np_to_tensor(self, np_data):
        return torch.tensor(np_data, dtype=torch.float32)
if __name__ == '__main__':
    env=gym.make('Pendulum-v0')
    a=agent()
    for b in range(a.epoch):
        observation = env.reset()
        print('第', b, '次遊戲')
        reward_sum=0
        while True:
            env.render()
            state=a.np_to_tensor(observation)
            action=a.Actor(state)
            print(action)
            observation, r, done, info = env.step(a.tensor_to_np(action))
            reward_sum+=r
            if done:
                print(reward_sum)
                break