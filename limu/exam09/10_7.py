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

add_norm = AddNorm([3, 4], 0.5)
add_norm.eval()
add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape

# print('1' * 100)
# print(result)

#@save
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
encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
encoder_blk.eval()
print(encoder_blk(X, valid_lens).shape)
