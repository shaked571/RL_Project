import numpy as np
import random
import gym
from collections import defaultdict
from algorithm import Algo


class Qlearning(Algo):
    state_bounds = [
                    (-1, 1),  # position X
                    (-1, 1),  # position Y
                    (-1, 1),  # velocity X
                    (-1, 1),  # velocity Y
                    (-1, 1),  # angel
                    (-1, 1),  # angular velocity
                    (0, 1),  # left leg touch ground
                    (0, 1)  # right leg touch ground
                    ]

    ALWAYS_DISCRETE_VALUES = {6, 7}

    def __init__(self, env, discrete_size_state, discrete_size_action):
        super().__init__(env, "qlearning")
        self.discrete_size_state = discrete_size_state
        self.discrete_size_action = discrete_size_action
        self.action_size = self.env.action_space.shape[0]
        self.q_table = defaultdict(lambda: np.zeros(tuple([self.discrete_size_action]*self.action_size)))
        self.state_bins = [np.linspace(min_val, max_val, self.discrete_size_state) for min_val, max_val in self.state_bounds]
        self.action_bin = np.linspace(-1, 1, self.discrete_size_action)
        self.GAMMA = 0.99
        self.ALPHA = 0.01
        self.INF = 1000
        self.epsilon = 0.5
        self.all_states = np.array([0] * len(self.state_bounds))

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
            print(f"Algo step number {i} got best score: {self.high_score}")

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
            for i in range(0, self.action_size):
                action += (random.randint(0, self.discrete_size_action - 1),)
        else:
            action = np.unravel_index(np.argmax(self.q_table[state]), self.q_table[state].shape)

        return action

    def convert_2_action(self, next_discrete_action):
        return tuple([self.action_bin[a] for a in next_discrete_action])


def main():
    env = gym.make("LunarLanderContinuous-v2")
    algo = Qlearning(env, 10, 5)
    algo.run_all_episodes()


if __name__ == '__main__':
    main()
