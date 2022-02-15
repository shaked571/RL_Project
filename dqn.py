import itertools

import gym
import numpy as np
import torch
from algorithm import Algo
from buffer import ReplayBuffer
from models import DQNModel
from utils import OrnsteinUhlenbeckActionNoise, hard_update

MAX_REPLAY_BUFFER = 100000


class DQN(Algo):

    def __init__(self,   env, action_space: dict, buffer, device, noise, apply_noise, discount_factor=0.99, update_rate=100, batch_size=64,
                 lr=0.001, epsilon=1, eps_end=0.01, eps_dec=5e-4, max_steps=1000):
        super().__init__(env, 'dqn')
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = action_space
        self.num_actions = len(self.action_space)
        self.device = device
        self.update_rate = update_rate
        self.batch_size = batch_size
        self.buffer = buffer
        self.discount_factor = discount_factor
        self.max_steps = max_steps
        self.noise = noise
        self.apply_noise = apply_noise

        self.step_counter = 0
        self.scores = []

        self.state_dim = env.observation_space.shape[0]

        self.Q_net = DQNModel(state_dim=self.state_dim, action_dim=self.num_actions, hidden_dim=256)
        self.Q_target = DQNModel(state_dim=self.state_dim, action_dim=self.num_actions, hidden_dim=256)
        self.optimizer = torch.optim.AdamW(self.Q_net.parameters(), self.lr)
        self.loss = torch.nn.HuberLoss()

    def policy(self, observation):
        if np.random.random() < self.epsilon:
            actions = list(self.action_space.keys())
            action = np.random.choice(actions)
            return action
        else:
            state = np.array([observation])
            actions = self.Q_net(state)
            action = torch.argmax(actions, dim=1).numpy()[0]
            return action

    def run_algo_step(self, i):
        print(f'EPISODE: {i}')
        state = self.env.reset()
        total_reward = 0

        for _ in range(self.max_steps):
            self.env.render()
            action = self.policy(state)
            next_state, reward, done, _ = self.env.step(self.action_space[action])
            next_state = state + self.noise.sample().numpy() if self.apply_noise else next_state
            total_reward += reward
            self.buffer.add(state, action, reward, next_state, done)
            state = next_state

            if self.buffer.len < self.batch_size:
                continue

            if self.step_counter % self.update_rate == 0:
                hard_update(self.Q_target, self.Q_net)

            state_batch, action_batch, reward_batch, new_state_batch, done_batch = self.buffer.sample(self.batch_size)

            # step
            q_predicted = self.Q_net(state_batch)
            q_next = self.Q_target(new_state_batch).detach().max(1)[0].unsqueeze(1)
            # q_max_next = torch.max(q_next, dim=1).values.detach().numpy()
            # q_target = np.copy(q_predicted.detach())

            # Get max predicted Q values (for next states) from target model
            # Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            # Compute Q targets for current states
            q_target = reward_batch + (self.discount_factor * q_next * (1 - done_batch))

            # for idx in range(done_batch.shape[0]):
            #     q_predicted[idx, int(action_batch[idx])] = reward_batch[idx] + self.discount_factor * q_max_next[idx] * (1 - int(done_batch[idx]))

            # train
            self.optimizer.zero_grad()
            self.Q_net.zero_grad()
            # output = self.Q_net(state_batch)
            out_loss = self.loss(q_target, q_predicted)
            out_loss.backward()
            self.optimizer.step()

            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
            self.step_counter += 1

        return total_reward


def main():
    seed = 2022

    # action_space = generate_action_spaces([0,0.3, 0.6, 0.9], [0, 0.6, -0.6, 0.9, -0.9])
    action_space = {i: v for i, v in enumerate(itertools.product([0, 0.3, 0.6, 0.9], [0, 0.6, -0.6, 0.9, -0.9]))}

    env = gym.make("LunarLanderContinuous-v2")
    env.action_space.np_random.seed(seed)
    # random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    noise = OrnsteinUhlenbeckActionNoise(action_dim=1, rng=rng, theta=0.005, sigma=0.005)
    buffer = ReplayBuffer(MAX_REPLAY_BUFFER, device, rng)

    algo = DQN(env, action_space, buffer, device, noise, apply_noise=True)
    algo.run_all_episodes()


if __name__ == '__main__':
    main()
