import torch
import torch.nn as nn


class Critic(nn.Module):

	def __init__(self, state_dim, action_dim):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of input action (int)
		:return:
		"""
		super(Critic, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim

		self.fcs1 = nn.Linear(state_dim, 256)
		self.fcs2 = nn.Linear(256, 128)

		self.fca1 = nn.Linear(action_dim, 128)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 32)
		self.fc4 = nn.Linear(32, 1)
		self.leaky_relu = nn.LeakyReLU(0.1)

		nn.init.xavier_uniform_(self.fcs1.weight)
		nn.init.xavier_uniform_(self.fcs2.weight)
		nn.init.xavier_uniform_(self.fca1.weight)
		nn.init.xavier_uniform_(self.fc2.weight)
		nn.init.xavier_uniform_(self.fc3.weight)
		nn.init.xavier_uniform_(self.fc4.weight)

	def forward(self, state, action):
		"""
		returns Value function Q(s,a) obtained from critic network
		:param state: Input state (Torch Variable : [n,state_dim] )
		:param action: Input Action (Torch Variable : [n,action_dim] )
		:return: Value function : Q(S,a) (Torch Variable : [n,1] )
		"""
		s1 = self.leaky_relu(self.fcs1(state))
		s2 = self.leaky_relu(self.fcs2(s1))
		a1 = self.leaky_relu(self.fca1(action))
		x = torch.cat((s2, a1), dim=1)

		x = self.leaky_relu(self.fc2(x))
		x = self.leaky_relu(self.fc3(x))
		x = self.fc4(x)

		return x


class Actor(nn.Module):

	def __init__(self, state_dim, action_dim, action_lim):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of output action (int)
		:return:
		"""
		super(Actor, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = action_lim

		self.fc1 = nn.Linear(state_dim, 256)
		self.fc2 = nn.Linear(256, 128)

		self.fc3 = nn.Linear(128, 64)
		self.fc4 = nn.Linear(64, action_dim)
		self.leaky_relu = nn.LeakyReLU(0.15)

		nn.init.xavier_uniform_(self.fc1.weight)
		nn.init.xavier_uniform_(self.fc2.weight)
		nn.init.xavier_uniform_(self.fc3.weight)
		nn.init.xavier_uniform_(self.fc4.weight)

	def forward(self, state):
		"""
		returns policy function Pi(s) obtained from actor network
		this function is a gaussian prob distribution for all actions
		with mean lying in (-1,1) and sigma lying in (0,1)
		The sampled action can , then later be rescaled
		:param state: Input state (Torch Variable : [n,state_dim] )
		:return: Output action (Torch Variable: [n,action_dim] )
		"""
		x = self.leaky_relu(self.fc1(state))
		x = self.leaky_relu(self.fc2(x))
		x = self.leaky_relu(self.fc3(x))
		action = torch.tanh(self.fc4(x))

		action = action * self.action_lim

		return action


class DQNModel(nn.Module):

	def __init__(self, state_dim, action_dim, hidden_dim):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of output action (int)
		:return:
		"""
		super(DQNModel, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.hidden_dim = hidden_dim

		self.fc1 = nn.Linear(state_dim, self.hidden_dim)
		self.fc2 = nn.Linear(self.hidden_dim, int(self.hidden_dim/2))
		self.fc3 = nn.Linear(int(self.hidden_dim / 2), action_dim)

		self.leaky_relu = nn.LeakyReLU(0.1)

		nn.init.xavier_uniform_(self.fc1.weight)
		nn.init.xavier_uniform_(self.fc2.weight)
		nn.init.xavier_uniform_(self.fc3.weight)

	def forward(self, state):
		"""
		returns policy function Pi(s) obtained from actor network
		this function is a gaussian prob distribution for all actions
		with mean lying in (-1,1) and sigma lying in (0,1)
		The sampled action can , then later be rescaled
		:param state: Input state (Torch Variable : [n,state_dim] )
		:return: Output action (Torch Variable: [n,action_dim] )
		"""
		x = self.leaky_relu(self.fc1(state))
		x = self.leaky_relu(self.fc2(x))
		action = self.fc3(x)

		return action

