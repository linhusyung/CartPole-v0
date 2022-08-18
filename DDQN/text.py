import gym
import torch
from network_torch import DQN_torch

class agent():
    def __init__(self):
        self.observation = env.reset()
        self.model=self.model = DQN_torch(state_dim=len(self.observation))
        self.epoch=20
        self.PATH='model/model_params.pth'
    def choose_action(self, out):
        return int(out.argmax().cpu().numpy())
    def np_to_tensor(self, np_data):
        return torch.tensor(np_data, dtype=torch.float32)
if __name__ == '__main__':
    env=gym.make('CartPole-v0')
    a=agent()
    for b in range(a.epoch):
        observation = env.reset()
        print('第', b, '次遊戲')
        while True:
            env.render()
            a.model.load_state_dict(torch.load(a.PATH))
            state=a.np_to_tensor(observation)
            out=a.model(state)
            action=a.choose_action(out)
            observation, r, done, info = env.step(action)
            if done:
                break