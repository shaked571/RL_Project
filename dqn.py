import itertools

import gym
import numpy as np
import torch
from algorithm import Algo
from buffer import ReplayBuffer
from models import DQNModel


class DQN(Algo):

    def __init__(self,   env, action_space,buffer, device,batch_size=64, lr=0.001, max_mem_size=100000, epsilon=1, eps_end=0.01,
                 eps_dec=5e-4):
        super().__init__(env, 'dqn')
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = action_space
        self.num_actions = len(self.action_space)
        self.device = device

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.Q_net = DQNModel(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=256)
        self.Q_target = DQNModel(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=256)

    def run_algo_step(self, i):

def main():
    seed = 2022

    # action_space = generate_action_spaces([0,0.3, 0.6, 0.9], [0, 0.6, -0.6, 0.9, -0.9])
    action_space = {i: v for i, v in enumerate(itertools.product(*[np.linspace(-0.8, 0.8, 5)] * 4))}

    env = gym.make("LunarLanderContinuous-v2")
    env.action_space.np_random.seed(seed)
    # random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_REPLAY_BUFFER = 100000
    buffer = ReplayBuffer(MAX_REPLAY_BUFFER, device, rng)

    algo = DQN(env,action_space, buffer, device)
    algo.run_all_episodes()

if __name__ == '__main__':
    main()
