# -*- coding: utf-8 -*-
# @Time : 2023/12/30 23:55
# @Author : nanji
# @Site : 
# @File : 68_p3.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
import math
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
# a = ffn(torch.ones((2, 3, 4)))
# print(a.shape)

ln = nn.LayerNorm(2)
bn = nn.BatchNorm1d(2)
X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)


# print('layer norm :', ln(X), '\n batch norm:', bn(X))


class AddNorm(nn.Module):
	def __init__(self, normalized_shape, dropout, **kwargs):
		super(AddNorm, self).__init__(**kwargs)
		self.dropout = nn.Dropout(dropout)
		self.ln = nn.LayerNorm(normalized_shape)

	def forward(self, X, Y):
		return self.ln(self.dropout(Y) + X)


add_norm = AddNorm([3, 4], 0.5)
add_norm.eval()


# print(add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape)
class EncoderBlock(nn.Module):
	def __init__(self, key_size, query_size, value_size, \
				 num_hidden, norm_shape, \
				 ffn_num_input, ffn_num_hiddens, \
				 num_heads, dropout, use_bias=False, **kwargs):
		super(EncoderBlock, self).__init__(**kwargs)
		self.attention = d2l.MultiHeadAttention(
			key_size, query_size,
			value_size, num_hidden,
			num_heads, dropout, use_bias
		)
		self.addnorm1 = AddNorm(norm_shape, dropout)
		self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, \
								   num_hidden)
		self.addnorm2 = AddNorm(norm_shape, dropout)

	def forward(self, X, valid_lens):
		Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
		return self.addnorm2(Y, self.ffn(Y))


X = torch.ones((2, 100, 24))
valid_lens = torch.tensor([3, 2])
encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
encoder_blk.eval()


# print(encoder_blk(X, valid_lens).shape)


class TransformerEncoder(d2l.Encoder):
	def __init__(self, vocab_size, key_size, query_size, \
				 value_size, num_hiddens, \
				 norm_shape, ffn_num_inputs, \
				 ffn_num_hiddens, num_heads, \
				 num_layers, dropout, use_bias=False, **kwargs
				 ):
		super(TransformerEncoder, self).__init__(**kwargs)
		self.num_hiddens = num_hiddens
		self.embedding = nn.Embedding(vocab_size, num_hiddens)
		self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
		self.blks = nn.Sequential()
		for i in range(num_layers):
			self.blks.add_module(
				'block' + str(i),
				EncoderBlock(key_size, query_size, \
							 value_size, num_hiddens, \
							 norm_shape, ffn_num_inputs, \
							 ffn_num_hiddens, num_heads, \
							 dropout, use_bias))

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


# print(encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape)

class DecoderBlock(nn.Module):
	'''解码器中第i个快 '''

	def __init__(self, key_size, query_size, value_size, \
				 num_hiddens, norm_shape, ffn_num_input, \
				 ffn_num_hiddens, num_heads, \
				 dropout, i, **kwargs):
		super(DecoderBlock, self).__init__(**kwargs)
		self.i = i
		self.attention1 = d2l.MultiHeadAttention(
			key_size, query_size, \
			value_size, num_hiddens, \
			num_heads, dropout
		)
		self.addnorm1 = AddNorm(norm_shape, dropout)
		self.attention2 = d2l.MultiHeadAttention(
			key_size, query_size,
			value_size, num_hiddens,
			num_heads, dropout
		)
		self.addnorm2 = AddNorm(norm_shape, dropout)
		self.ffn = PositionWiseFFN(ffn_num_input, \
								   ffn_num_hiddens, \
								   num_hiddens)
		self.addnorm3 = AddNorm(norm_shape, dropout)

	def forward(self, X, state):
		enc_outputs, enc_valid_lens = state[0], state[1]
		if state[2][self.i] is None:
			key_values = X
		else:
			key_values = torch.cat((state[2][self.i], X), aixs=1)
		state[2][self.i] = key_values
		if self.training:
			batch_size, num_steps, _ = X.shape
			dec_valid_lens = torch.arange(
				1, num_steps + 1, device=X.device).repeat(batch_size, 1)
		else:
			dec_valid_lens = None
		X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
		Y = self.addnorm1(X, X2)
		Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
		Z = self.addnorm2(Y, Y2)
		return self.addnorm3(Z, self.ffn(Z)), state


decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
decoder_blk.eval()
X = torch.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
print(decoder_blk(X, state)[0].shape)
