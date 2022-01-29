import numpy as np
import random
import math
import gym
from collections import defaultdict
from algorithm import Algo


class Qlearning(Algo):
    state_bounds = [(-1, math.pi),  # 0 hull_angle
                    (-0.25, 0.25),  # 1 hull_angularVelocity
                    (-1, 1),  # 2 vel_x
                    (-1, 1),  # 3 vel_y
                    (-0.5, 0.5),  # 4 hip_joint_1_angle
                    (-2.5, 2.5),  # 5 hip_joint_1_speed
                    (-0.5, 1.5),  # 6 knee_joint_1_angle
                    (-3, 3),  # 7 knee_joint_1_speed
                    (0, 1),  # 8 leg_1_ground_contact_flag
                    (-0.8, 1),  # 9 hip_joint_2_angle
                    (-2, 2),  # 10 hip_joint_2_speed
                    (-1, 0.5),  # 11 knee_joint_2_angle
                    (-3, 3),  # 12 knee_joint_2_speed
                    (0, 1),  # 13 leg_2_ground_contact_flag
                    (0, 1),  # lidar 1
                    (0, 1),  # lidar 2
                    (0, 1),  # lidar 3
                    (0, 1),  # lidar 4
                    (0, 1),  # lidar 5
                    (0, 1),  # lidar 6
                    (0, 1),  # lidar 7
                    (0, 1),  # lidar 8
                    (0, 1),  # lidar 9
                    (0, 1)  # lidar 10
                    ]

    ALWAYS_DISCRETE_VALUES = {8, 13}

    def __init__(self, env, discrete_size):
        super().__init__(env)
        self.discrete_size = discrete_size
        self.q_table = defaultdict(lambda: np.zeros((discrete_size, discrete_size, discrete_size, discrete_size)))
        self.state_bins = [np.linspace(min_val, max_val, self.discrete_size) for min_val, max_val in self.state_bounds]
        self.action_bin = np.linspace(-1, 1, self.discrete_size)
        self.GAMMA = 0.99
        self.ALPHA = 0.01
        self.INF = 1000
        self.epsilon = 0.5
        self.all_states = np.array([0] * 24)

    def update_q_table(self, state, action, reward, nextState=None):

        current = self.q_table[state][action]
        qNext = np.max(self.q_table[nextState]) if nextState is not None else 0
        target = reward + (self.GAMMA * qNext)
        new_value = current + (self.ALPHA * (target - current))
        return new_value

    def epsilon_decay(self):
        return max(self.epsilon * 0.99, 0.1)

    def run_algo_step(self, i):

        print("Episode #: ", i)
        env_state = self.env.reset()
        state = self.discretize_state(env_state)
        total_reward = 0
        while True:
            n_discrete_action = self.get_next_disc_action(state)
            next_real_action = self.convert_2_action(n_discrete_action)
            next_state, reward, done, info = self.env.step(next_real_action)
            next_state = self.discretize_state(next_state)
            total_reward += reward
            self.q_table[state][n_discrete_action] = self.update_q_table(state, n_discrete_action, reward, next_state)
            state = next_state

            if self.DEBUG:
                self.all_states = np.vstack((self.all_states, next_state))

            if done:
                if self.DEBUG:
                    self.print_states_vals(next_state)
                break
        if total_reward > self.high_score:
            self.high_score = total_reward
        self.epsilon_decay()
        return total_reward

    def print_states_vals(self, next_state):
        print("last state")
        print(next_state)
        print('max')
        print(self.all_states.max(axis=0))
        print('min')
        print(self.all_states.min(axis=0))

    def discretize_state(self, states):
        discrete_stats = []
        for i, state in enumerate(states):
            if i in self.ALWAYS_DISCRETE_VALUES:
                discrete_stats.append(state)
            else:
                dis_state = np.digitize(state, self.state_bins[i])
                discrete_stats.append(dis_state)
        return tuple(discrete_stats)

    def get_next_disc_action(self, state):
        if np.random.random() < self.epsilon:
            action = ()
            for i in range(0, 4):
                action += (random.randint(0, self.discrete_size - 1),)
        else:
            action = np.unravel_index(np.argmax(self.q_table[state]), self.q_table[state].shape)

        return action

    def convert_2_action(self, next_discrete_action):
        return tuple([self.action_bin[a] for a in next_discrete_action])


def main():
    env = gym.make("BipedalWalker-v3")
    algo = Qlearning(env, 10)
    algo.run_all_episodes()


if __name__ == '__main__':
    main()
