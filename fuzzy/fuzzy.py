import gym
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class fuzzy():
    def __init__(self):
        self.cart_pos = ctrl.Antecedent(np.arange(-4.8, 4.9, 0.01), 'cart_position')
        self.pole_angle = ctrl.Antecedent(np.arange(-24, 24.1, 0.01), 'pole_angle')
        self.action = ctrl.Consequent(np.arange(0, 2, 1), 'action')
        self.fuzzy_ctrl = None

    def create_fuzzy_controller(self):
        self.cart_pos['left'] = fuzz.trimf(self.cart_pos.universe, [-4.8, -4.8, 0.1])
        self.cart_pos['right'] = fuzz.trimf(self.cart_pos.universe, [-0.1, 4.8, 4.8])

        self.pole_angle['left'] = fuzz.trimf(self.pole_angle.universe, [-24, -24, 0])
        self.pole_angle['center'] = fuzz.trimf(self.pole_angle.universe, [-24, 0, 24])
        self.pole_angle['right'] = fuzz.trimf(self.pole_angle.universe, [0, 24, 24])

        self.action['left'] = fuzz.trimf(self.action.universe, [0, 0, 0.5])
        self.action['right'] = fuzz.trimf(self.action.universe, [0.5, 1, 1])

        rule1 = ctrl.Rule(self.cart_pos['left'] & self.pole_angle['left'], self.action['left'])
        rule2 = ctrl.Rule(self.cart_pos['left'] & self.pole_angle['center'], self.action['right'])
        rule3 = ctrl.Rule(self.cart_pos['left'] & self.pole_angle['right'], self.action['right'])

        rule4 = ctrl.Rule(self.cart_pos['right'] & self.pole_angle['left'], self.action['left'])
        rule5 = ctrl.Rule(self.cart_pos['right'] & self.pole_angle['center'], self.action['left'])
        rule6 = ctrl.Rule(self.cart_pos['right'] & self.pole_angle['right'], self.action['right'])

        self.fuzzy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])

    def get_action(self, observation):
        ctrl_action = ctrl.ControlSystemSimulation(self.fuzzy_ctrl)
        ctrl_action.input['cart_position'] = 0.5 * observation[0]
        ctrl_action.input['pole_angle'] = 0.8 * np.degrees(observation[2])
        ctrl_action.compute()
        action_value = ctrl_action.output['action']
        # print(action_value)
        if action_value >= 0.5:
            return 1
        else:
            return 0


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    fuzzy = fuzzy()
    fuzzy.create_fuzzy_controller()

    for b in range(10):
        observation = env.reset()
        print('第', b, '次遊戲')
        reward = 0
        while True:
            env.render()
            action_value = fuzzy.get_action(observation)
            observation, r, done, info = env.step(action_value)
            reward += r
            if done:
                print(reward)
                break
