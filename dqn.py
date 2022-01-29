import gym
import numpy as np
import torch
from algorithm import Algo
from models import DQNModel


class DQN(Algo):

    def __init__(self, gamma, epsilon, lr, batch_size, env, max_mem_size=100000, eps_end=0.05,
                 eps_dec=5e-4):
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.Q_eval = DQNModel(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=256)
        self.Q_next = DQNModel(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=256)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation]).float().to(self.Q_eval.device)
            actions = self.Q_eval(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)  # no duplicates indexes

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        # get the q value of every sample of the action played
        q_eval = self.Q_eval(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]  # take max q value for every sample

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min


if __name__ == '__main__':
    env = gym.make("BipedalWalker-v3")
    input_dims = len(env.reset())  # how many elements does the state representation have?
    agent = DQN(env=env, gamma=0.99, epsilon=1.0, batch_size=4, eps_end=0.01,
                     lr=0.003)
    scores, avg_scores, eps_history = [], [], []


    #TODO -refactor!!!
    epochs = 500

    for epoch in range(epochs):
        score = 0
        done = False
        state_old = env.reset()
        # print(state_old[0].type)
        while not done:  # iterating over every timestep (state)
            env.render()
            action = agent.choose_action(state_old)
            state_new, reward, done, info = env.step(action)
            score += reward

            agent.store_transition(state_old, action, reward, state_new, done)
            agent.learn()
            state_old = state_new

        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)

        print("epoch: ", epoch, "score: %.2f " % score, "avg_score: %.2f " % avg_score, "epsilon: %.2f" % agent.epsilon)

    env.close()
