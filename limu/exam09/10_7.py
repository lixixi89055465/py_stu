# -*- coding: utf-8 -*-
# @Time : 2024/1/25 10:43
# @Author : nanji
# @Site : 
# @File : 10_7.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
import math
from torch import nn
from d2l import torch as d2l


class PositionWiseFFN(nn.Module):
	"""基于位置的前馈网络"""

	def __init__(self, ffn_num_input, ffn_num_hiddens, \
				 ffn_num_outputs, **kwargs):
		super(PositionWiseFFN, self).__init__(**kwargs)
		self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
		self.relu = nn.ReLU()
		self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

	def forward(self, X):
		return self.dense2(self.relu(self.dense1(X)))


# ffn = PositionWiseFFN(4, 4, 8)
# ffn.eval()


# result=ffn(torch.ones((2, 3, 4)))
# print(result[0])

# ln = nn.LayerNorm(2)
# bn = nn.BatchNorm1d(2)
# X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
# print('layer norm :', ln(X), '\n batch norm :', bn(X))

class AddNorm(nn.Module):
	"""残差连接后进行层规范化"""

	def __init__(self, normalized_shape, dropout, **kwargs):
		super(AddNorm, self).__init__(**kwargs)
		self.dropout = nn.Dropout(dropout)
		self.ln = nn.LayerNorm(normalized_shape)

	def forward(self, X, Y):
		return self.ln(self.dropout(Y) + X)


# add_norm = AddNorm([3, 4], 0.5)
# add_norm.eval()
# add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape


# print('1' * 100)
# print(result)

# @save
class EncoderBlock(nn.Module):
	"""Transformer编码器块"""

	def __init__(self, key_size, query_size, value_size, num_hiddens,
				 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
				 dropout, use_bias=False, **kwargs):
		super(EncoderBlock, self).__init__(**kwargs)
		self.attention = d2l.MultiHeadAttention(
			key_size, query_size, value_size, num_hiddens, num_heads, dropout,
			use_bias)
		self.addnorm1 = AddNorm(norm_shape, dropout)
		self.ffn = PositionWiseFFN(
			ffn_num_input, ffn_num_hiddens, num_hiddens)
		self.addnorm2 = AddNorm(norm_shape, dropout)

	def forward(self, X, valid_lens):
		Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
		return self.addnorm2(Y, self.ffn(Y))


X = torch.ones((2, 100, 24))
valid_lens = torch.tensor([3, 2])
# encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
# encoder_blk.eval()
# print(encoder_blk(X, valid_lens).shape)

class TransformerEncoder(d2l.Encoder):
	def __init__(self, vocab_size, key_size, query_size, value_size, \
				 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, \
				 num_heads, num_layers, dropout, use_bias=False, **kwargs):
		super(TransformerEncoder, self).__init__(**kwargs)
		self.num_hiddens = num_hiddens
		self.embedding = nn.Embedding(vocab_size, num_hiddens)
		self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
		self.blks = nn.Sequential()
		for i in range(num_layers):
			self.blks.add_module('block' + str(i),
								 EncoderBlock(key_size, query_size, value_size, num_hiddens, \
											  norm_shape, ffn_num_input, ffn_num_hiddens, \
											  num_heads, dropout, use_bias))

	def forward(self, X, valid_lens, *args):
		X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
		self.attention_weights = [None] * len(self.blks)
		for i, blk in enumerate(self.blks):
			X = blk(X, valid_lens)
			self.attention_weights[i] = blk.attention.attention.attention_weights
		return X


encoder = TransformerEncoder(
	200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5
)
encoder.eval()
print(encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape)
