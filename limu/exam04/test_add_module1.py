# -*- coding: utf-8 -*-
# @Time    : 2023/12/10 下午5:17
# @Author  : nanji
# @Site    : 
# @File    : test_add_module.py
# @Software: PyCharm 
# @Comment :
import torchvision
import torch
import torch.nn as nn


class NeuralNetWork(nn.Module):
	def __init__(self, layer_num=2):
		super(NeuralNetWork, self).__init__()
		self.layers = [nn.Linear(28 * 28, 28 * 28) for _ in range(layer_num)]
		for i, layer in enumerate(self.layers):
			self.add_module(f'layer_{i}', layer)
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(28 * 28, 512)
			, nn.ReLU()
		)

	def forward(self, X):
		for layer in self.layers:
			X = layer(X)
		logits = self.linear_relu_stack(X)
		return logits


# net = NeuralNetWork()
# print(list(net.children()))
# print('0' * 100)

# net2 = NeuralNetWork(2)
# print(list(net2.children()))

print('1' * 100)
model = NeuralNetWork(4)
for index, item in enumerate(model.children()):
	print(index, item)
