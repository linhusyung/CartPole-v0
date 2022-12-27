import gym
import numpy as np


class Q_learning():
    def __init__(self):
        self.env = gym.make("FrozenLake-v1", desc=['SFFHF', 'FHFFF', 'FFHFF', 'HFFHF', 'HFFFG'], is_slippery=False)
        action_space_size = self.env.action_space.n
        state_space_size = self.env.observation_space.n
        self.qtable = np.zeros((state_space_size, action_space_size))
        self.eps = 1
        self.gamma = 0.99
        self.lr = 0.9

    def updata_Qtable(self, state, state_next_, action, reward):
        self.qtable[state, action] = self.qtable[state, action] + self.lr * (
                reward + self.gamma * np.max(self.qtable[state_next_]) - self.qtable[state, action])

    def choose_action(self, state):
        if np.random.random() > self.eps:
            return np.random.randint(0, 4)
        else:
            return np.argmax(self.qtable[state])


if __name__ == '__main__':
    Q = Q_learning()
    print(Q.qtable.shape)

    for _ in range(10000):
        obs = Q.env.reset()
        while True:
            # Q.env.render()
            state = obs
            action = Q.choose_action(state)
            obs, reward, done, info = Q.env.step(action)

            # if reward==0 and done==0:
            #     reward=-0.1
            # if reward==0 and done==1:
            #     reward=-1

            Q.updata_Qtable(state, obs, action, reward)

            if Q.eps > 0.1:
                Q.eps *= 0.9

            if done:
                break

    # text
    print('開始測試')
    success = 0
    for _ in range(100):
        obs = Q.env.reset()
        while True:
            Q.env.render()
            state = obs
            action = np.argmax(Q.qtable[state])
            obs, reward, done, info = Q.env.step(action)
            if done:
                if reward != 0:
                    success += 1
                    print('reward', reward, 'done', done)
                break
    print('成功率',success/100)