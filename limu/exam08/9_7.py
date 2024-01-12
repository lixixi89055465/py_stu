# -*- coding: utf-8 -*-
# @Time : 2024/1/11 17:44
# @Author : nanji
# @Site : 
# @File : 9_7.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l
import collections
import math


class Seq2SeqEncoder(d2l.Encoder):
	def __init__(self, vocab_size, embed_size, num_hidden, num_layers, \
				 dropout=0, **kwargs):
		super(Seq2SeqEncoder, self).__init__(**kwargs)
		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.rnn = nn.GRU(embed_size, num_hidden, num_layers, \
						  dropout=dropout)
