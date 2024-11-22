# -*- coding: utf-8 -*-
# @Time    : 2024/11/22 下午9:51
# @Author  : nanji
# @Site    : https://blog.csdn.net/m0_57317650/article/details/134869466
# @File    : testIdentityNetwork.py
# @Software: PyCharm 
# @Comment :
# 定义一个包含Identity曾的简单忘了
import torch
import torch.nn as nn


class IdentityNetwork(nn.Module):
	def __init__(self):
		super(IdentityNetwork, self).__init__()
		self.layer1 = nn.Linear(10, 5)
		self.identity = nn.Identity()

	def forward(self, x):
		x = self.layer1(x)
		x_identity = self.identity(x)
		return x, x_identity


model = IdentityNetwork()
input_tensor = torch.randn(2, 10)
output, output_identity = model(input_tensor)
print('Output form model')
print(output)
print('\n Output from the identity layer:')
print(output)
