# -*- coding: utf-8 -*-
# @Time : 2024/1/24 11:08
# @Author : nanji
# @Site : 
# @File : 10_5.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l


class MultiHeadAttention(nn.Module):
	def __init__(self, key_size, query_size, value_size, \
				 num_hiddens, num_heads, dropout, bias=False, **kwargs):
		super(MultiHeadAttention, self).__init__(**kwargs)
		self.num_heads = num_heads
		self.attention = d2l.DotProductAttention(dropout)
		self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
		self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
		self.W_v = nn.Linear(value_size, num_hiddens, bias=False)
		self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=False)

	def forward(self, queries, keys, values, valid_lens):
		queries = transpose_qkv(self.W_q(queries), self.num_heads)
		keys = transpose_qkv(self.W_k(keys), self.num_heads)
		values = transpose_qkv(self.W_v(values), self.num_heads)
		if valid_lens is not None:
			valid_lens = torch.repeat_interleave(
				valid_lens, repeats=self.num_heads, dim=0
			)
		output = self.attention(queries, keys, values, valid_lens)
		output_concat = transpose_output(output, self.num_heads)
		return self.W_o(output_concat)


def transpose_qkv(X, num_heads):
	X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
	X = X.permute(0, 2, 1, 3)
	return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
	X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
	X = X.permute(0, 2, 1, 3)
	return X.reshape(X.shape[0], X.shape[1], -1)


num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, \
							   num_hiddens, \
							   num_hiddens, \
							   num_hiddens, \
							   num_heads, \
							   0.5)
attention.eval()

batch_size, num_queries = 2, 4
num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
result = attention(X, Y, Y, valid_lens).shape
print('1' * 100)
print(result)
