import torch
import torch.nn.functional as F
from algorithm import Algo
import utils
from model import Actor, Critic
import numpy as np
import gym
import buffer

BATCH_SIZE = 128
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001
MAX_REPLAY_BUFFER = 1000000


class ActorCritic(Algo):
	MAX_STEPS = 1000

	def __init__(self, ram, env):
		"""
		:param ram: replay memory buffer object
		"""
		super().__init__(env)
		self.state_dim = env.observation_space.shape[0]
		self.action_dim = env.action_space.shape[0]
		self.action_lim = env.action_space.high[0]

		print(f'State Dimensions: {self.state_dim}')
		print(f'Action Dimensions: {self.action_dim}')
		print(f'Action Max: {self.action_lim}')

		self.ram = ram
		self.iter = 0
		self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)

		self.actor = Actor(self.state_dim, self.action_dim, self.action_lim)
		self.target_actor = Actor(self.state_dim, self.action_dim, self.action_lim)
		self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), LEARNING_RATE)

		self.critic = Critic(self.state_dim, self.action_dim)
		self.target_critic = Critic(self.state_dim, self.action_dim)
		self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), LEARNING_RATE)

		utils.hard_update(self.target_actor, self.actor)
		utils.hard_update(self.target_critic, self.critic)

	def get_exploitation_action(self, state):
		"""
		gets the action from target actor added with exploration noise
		:param state: state (Numpy array)
		:return: sampled action (Numpy array)
		"""
		state = torch.from_numpy(state)
		action = self.target_actor(state).detach()
		return action.data.numpy()

	def get_exploration_action(self, state):
		"""
		gets the action from actor added with exploration noise
		:param state: state (Numpy array)
		:return: sampled action (Numpy array)
		"""
		state = torch.from_numpy(state)
		action = self.actor(state).detach()
		new_action = action.data.numpy() + (self.noise.sample() * self.action_lim)
		a = action.data.numpy()
		return new_action

	def optimize(self):
		"""
		Samples a random batch from replay memory and performs optimization
		:return:
		"""
		s1, a1, r1, s2 = self.ram.sample(BATCH_SIZE)

		# s1 = torch.from_numpy(s1)
		# a1 = torch.from_numpy(a1)
		# r1 = torch.from_numpy(r1)
		# s2 = torch.from_numpy(s2)

		# ---------------------- optimize critic ----------------------
		# Use target actor exploitation policy here for loss evaluation
		a2 = self.target_actor(s2).detach()
		next_val = torch.squeeze(self.target_critic(s2, a2).detach())
		# y_exp = r + gamma*Q'( s2, pi'(s2))
		y_expected = r1 + GAMMA*next_val
		# y_pred = Q( s1, a1)
		y_predicted = torch.squeeze(self.critic(s1, a1))
		loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
		# loss_critic = torch.nn.SmoothL1Loss(y_predicted, y_expected)
		self.critic_optimizer.zero_grad()
		loss_critic.backward()
		self.critic_optimizer.step()

		# ---------------------- optimize actor ----------------------
		pred_a1 = self.actor.forward(s1)
		loss_actor = -1 * torch.sum(self.critic(s1, pred_a1))
		self.actor_optimizer.zero_grad()
		loss_actor.backward()
		self.actor_optimizer.step()

		utils.soft_update(self.target_actor, self.actor, TAU)
		utils.soft_update(self.target_critic, self.critic, TAU)

	def save_models(self, episode_count):
		"""
		saves the target actor and critic models
		:param episode_count: the count of episodes iterated
		:return:
		"""
		torch.save(self.target_actor.state_dict(), 'Models/' + str(episode_count) + '_actor.pt')
		torch.save(self.target_critic.state_dict(), 'Models/' + str(episode_count) + '_critic.pt')
		print('Models saved successfully')

	def load_models(self, episode):
		"""
		loads the target actor and critic models, and copies them onto actor and critic models
		:param episode: the count of episodes iterated (used to find the file name)
		:return:
		"""
		self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
		self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
		utils.hard_update(self.target_actor, self.actor)
		utils.hard_update(self.target_critic, self.critic)
		print('Models loaded succesfully')

	def run_algo_step(self, i):
		observation = self.env.reset()
		print(f'EPISODE: {i}')
		total_reward = 0

		while True:
			self.env.render()
			state = np.float32(observation)
			action = self.get_exploration_action(state)
			new_observation, reward, done, info = self.env.step(action)
			total_reward += reward

			if not done:
				new_state = np.float32(new_observation)
				self.ram.add(state, action, reward, new_state)

			observation = new_observation

			# perform optimization
			self.optimize()

			if done:
				break

		if i % 100 == 0:
			self.save_models(i)

		if total_reward > self.high_score:
			self.high_score = total_reward
		return total_reward


def main():
	env = gym.make("BipedalWalker-v3")
	ram = buffer.MemoryBuffer(MAX_REPLAY_BUFFER)
	algo = ActorCritic(ram, env)
	algo.run_all_episodes()


if __name__ == '__main__':
	main()
