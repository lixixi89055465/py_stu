# -*- coding: utf-8 -*-
# @Time : 2024/1/5 16:37
# @Author : nanji
# @Site : https://zh-v2.d2l.ai/chapter_recurrent-neural-networks/rnn-concise.html
# @File : 8_6.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

seed = 415
os.environ['PYTHONHASHSEED'] = str(seed)
# np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)
print(rnn_layer)

state = torch.zeros((1, batch_size, num_hiddens))
print('0' * 100)
print(state.shape)
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
print('1' * 100)
print(Y.shape, state_new.shape)


class RNNModel(nn.Module):
	def __init__(self, rnn_layer, vocab_size, **kwargs):
		super(RNNModel, self).__init__(**kwargs)
		self.rnn = rnn_layer
		self.vocab_size = vocab_size
		self.num_hiddens = vocab_size
		self.num_hiddens = num_hiddens
		if not self.rnn.bidirectional:
			self.num_directions = 1
			self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
		else:
			self.num_directions = 2
			self.liear = nn.Linear(self.num_directions * self.num_hiddens, self.vocab_size)

	def forward(self, inputs, state):
		X = F.one_hot(inputs, self.vocab_size)
		X = X.to(torch.float32)
		Y, state = self.rnn(X, state)
		output = self.linear(Y.reshape((-1, Y.shape[-1])))
		return output, state

	def begin_state(self, device, batch_size=1):
		if not isinstance(self.rnn, nn.LSTM):
			return torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device)
		else:
			return (
				torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device), \
				torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device),
			)


device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
print(d2l.predict_ch8('time traveller', 10, net, vocab, device))
