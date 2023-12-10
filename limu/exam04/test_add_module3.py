# -*- coding: utf-8 -*-
# @Time    : 2023/12/10 下午5:47
# @Author  : nanji
# @Site    : 
# @File    : test_add_module3.py
# @Software: PyCharm 
# @Comment :
from torch import nn


class Net1(nn.Module):
	def __init__(self):
		super(Net1, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 3)
		self.add_module('conv2', nn.Conv2d(6, 12, 3))
		self.conv2 = nn.Conv2d(12, 24, 3)

	def forward(self, X):
		X = self.conv1(X)
		X = self.conv2(X)
		X = self.conv3(X)
		return X


model = Net1()
print(model)
