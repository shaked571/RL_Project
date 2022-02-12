import numpy as np
import random
import torch
from collections import deque


class ReplayBuffer:

	def __init__(self, size, device):
		self.buffer = deque(maxlen=size)
		self.maxSize = size
		self.device = device
		self.len = 0

	def sample(self, batch_size):
		"""
		samples a random batch from the replay memory buffer
		:param batch_size: batch size
		:return: batch (numpy array)
		"""
		batch_size = min(batch_size, self.len)
		batch = random.sample(self.buffer, batch_size)

		cur_state = torch.tensor(np.float32([arr[0] for arr in batch])).to(self.device)
		action = torch.tensor(np.float32([arr[1] for arr in batch])).to(self.device)
		reward = torch.tensor(np.float32([arr[2] for arr in batch])).to(self.device)
		next_state = torch.tensor(np.float32([arr[3] for arr in batch])).to(self.device)

		return cur_state, action, reward, next_state

	def add(self, cur_state, action, reward, next_state):
		"""
		adds a particular transaction in the memory buffer
		:param cur_state: current state
		:param action: action taken
		:param reward: reward received
		:param next_state: next state
		:return:
		"""
		self.len += 1
		if self.len > self.maxSize:
			self.len = self.maxSize
		self.buffer.append((cur_state, action, reward, next_state))
