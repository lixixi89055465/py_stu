# -*- coding: utf-8 -*-
# @Time : 2024/1/25 10:43
# @Author : nanji
# @Site : 
# @File : 10_7.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l


class PositionWiseFFN(nn.Module):
	def __init__(self, ffn_num_input, ffn_num_hiddens, \
				 ffn_num_outputs, **kwargs):
		super(PositionWiseFFN, self).__init__(**kwargs)
		self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
		self.relu = nn.ReLU()
		self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_hiddens)

	def forward(self, X):
		return self.dense2(self.relu(self.dense1(X)))


ffn = PositionWiseFFN(4, 4, 8)
ffn.eval()


# result=ffn(torch.ones((2, 3, 4)))
# print(result[0])

# ln = nn.LayerNorm(2)
# bn = nn.BatchNorm1d(2)
# X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
# print('layer norm :', ln(X), '\n batch norm :', bn(X))

class AddNorm(nn.Module):
	def __init__(self, normalized_shape, dropout, **kwargs):
		super(AddNorm, self).__init__(**kwargs)
		self.dropout = nn.Dropout(dropout)
		self.ln = nn.LayerNorm(normalized_shape)

	def forward(self, X, Y):
		return self.ln(self.dropout(Y) + X)


add_norm = AddNorm([3, 4], 0.5)
add_norm.eval()
result = add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4)))


# print('1' * 100)
# print(result)

class EncoderBlock(nn.Module):
	def __init__(self, key_size, query_size, value_size, num_hiddens, \
				 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, \
				 dropout, use_bias=False, **kwargs):
		super(EncoderBlock, self).__init__(**kwargs)
		self.attention = d2l.MultiHeadAttention(
			key_size, query_size, value_size, \
			num_hiddens, num_heads, dropout, \
			use_bias)
		self.addnorm1 = AddNorm(norm_shape, dropout)
