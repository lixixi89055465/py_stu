import torchvision
import torch
import torch.nn  as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import itertools

initial_lr = 0.1


class model(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)

	def forward(self, x):
		pass


net_1 = model()
optimizer_1 = torch.optim.Adam(net_1.parameters(), lr=initial_lr)
schedualer_1 = ReduceLROnPlateau(optimizer_1, mode='min', factor=0.1, patience=2)
print('初始化的学习率', optimizer_1.defaults['lr'])
for epoch in range(1, 11):
	test = 2
	optimizer_1.zero_grad()
	optimizer_1.step()
	print(f'第{epoch}个epoch的学习率，{optimizer_1.param_groups[0]["lr"]}')
	schedualer_1.step(test)
