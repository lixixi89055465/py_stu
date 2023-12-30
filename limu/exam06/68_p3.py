# -*- coding: utf-8 -*-
# @Time : 2023/12/30 23:55
# @Author : nanji
# @Site : 
# @File : 68_p3.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l


class PositionWiseFFN(nn.Module):
	def __init__(self, ffn_num_input, ffn_num_hiddens, \
				 ffn_num_output, \
				 **kwargs):
		super(PositionWiseFFN, self).__init__(**kwargs)
		self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
		self.relu = nn.ReLU()
		self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_output)

	def forward(self, X):
		return self.dense2(self.relu(self.dense1(X)))


ffn = PositionWiseFFN(4, 4, 8)
ffn.eval()
a = ffn(torch.ones((2, 3, 4)))
print(a.shape)
