import torchvision
import torch
import torch.nn  as nn
from torch.optim.lr_scheduler import LambdaLR
import itertools

initial_lr = 0.1


class model(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, )
		self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, )

	def forward(self, x):
		pass


net_1 = model()
net_2 = model()
optimizer_1 = torch.optim.Adam(net_1.parameters(), lr=initial_lr)
print('************* optimzer***********')
print(optimizer_1.defaults)
print('0' * 100)
# print(len(optimizer_1.param_groups))
print('1' * 100)
print(optimizer_1.param_groups[0].keys())
print('2' * 100)
optimizer_2 = torch.optim.Adam([*net_1.parameters(), *net_2.parameters()], \
							   lr=initial_lr)
print(optimizer_2.defaults)
print('3' * 100)
print(optimizer_2.param_groups[0].keys())
print('4' * 100)
optimizer_3 = torch.optim.Adam([{'params': net_1.parameters()}, {'params': net_2.parameters()}], lr=initial_lr)
print(optimizer_3.defaults)
print('5' * 100)
print(optimizer_3.param_groups)
print('6' * 100)
print(optimizer_3.param_groups[0].keys())
