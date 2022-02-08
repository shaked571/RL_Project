# import numpy as np
# import torch
# import shutil
# import torch.autograd as Variable
#
#
# def soft_update(target, source, tau):
# 	"""
# 	Copies the parameters from source network (x) to target network (y) using the below update
# 	y = TAU*x + (1 - TAU)*y
# 	:param target: Target network (PyTorch)
# 	:param source: Source network (PyTorch)
# 	:return:
# 	"""
# 	for target_param, param in zip(target.parameters(), source.parameters()):
# 		target_param.data.copy_(
# 			target_param.data * (1.0 - tau) + param.data * tau
# 		)
#
#
# def hard_update(target, source):
# 	"""
# 	Copies the parameters from source network to target network
# 	:param target: Target network (PyTorch)
# 	:param source: Source network (PyTorch)
# 	:return:
# 	"""
# 	for target_param, param in zip(target.parameters(), source.parameters()):
# 			target_param.data.copy_(param.data)
#
#
# def save_training_checkpoint(state, is_best, episode_count):
# 	"""
# 	Saves the models, with all training parameters intact
# 	:param state:
# 	:param is_best:
# 	:param filename:
# 	:return:
# 	"""
# 	filename = str(episode_count) + 'checkpoint.path.rar'
# 	torch.save(state, filename)
# 	if is_best:
# 		shutil.copyfile(filename, 'model_best.pth.tar')
#
#
# # Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
# class OrnsteinUhlenbeckActionNoise:
#
# 	def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
# 		self.action_dim = action_dim
# 		self.mu = mu
# 		self.theta = theta
# 		self.sigma = sigma
# 		self.X = np.ones(self.action_dim) * self.mu
#
# 	def reset(self):
# 		self.X = np.ones(self.action_dim) * self.mu
#
# 	def sample(self):
# 		dx = self.theta * (self.mu - self.X)
# 		dx = dx + self.sigma * np.random.randn(len(self.X))
# 		self.X = self.X + dx
# 		return self.X
#
#
# # use this to plot Ornstein Uhlenbeck random motion
# if __name__ == '__main__':
# 	ou = OrnsteinUhlenbeckActionNoise(1)
# 	states = []
# 	for i in range(100000):
# 		states.append(ou.sample())
# 	import matplotlib.pyplot as plt
#
# 	plt.plot(states)
# 	plt.show()
#
#
import numpy as np
import torch
from matplotlib import pyplot as plt

# model = torch.nn.Linear(2, 1)
# optimizer = torch.optim.SGD(model.parameters(), lr=1000)
# lambda1 = lambda epoch: 0.999 ** epoch
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
# # torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
#
# lrs = []
#
# for i in range(1000):
#     optimizer.step()
#     lrs.append(optimizer.param_groups[0]["lr"])
# #     print("Factor = ", round(0.65 ** i,3)," , Learning Rate = ",round(optimizer.param_groups[0]["lr"],3))
#     scheduler.step()

import pandas as pd
score_graph = plt.figure()
yval = []
for i in range(3000):
    norm = i/200
    v= -180 + norm *40+ np.random.uniform(-40 - norm , 40+norm )
    if v > 300:
        v = 300
    if v< -200:
        v = -200
    yval.append(v)
xval= np.arange(0, 3000) #np.random.uniform(-200,300 ,3000)
yval = np.array(yval).reshape((3000))
sub_plot = score_graph.add_subplot()
plt.xlabel("Episode #")
plt.ylabel("Score")
plt.title("Scores vs Episode")
plot_line, = sub_plot.plot(xval, yval)
sub_plot.set_xlim([0, 3000])
sub_plot.set_ylim([-220, 400])

x = pd.Series(yval, index=xval).rolling(100, min_periods=1).mean()
sub_plot.plot(xval, x, "--k")

plt.show()