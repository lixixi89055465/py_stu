# -*- coding: utf-8 -*-
# @Time    : 2023/12/19 23:20
# @Author  : nanji
# @Site    : 
# @File    : 55_p1.py
# @Software: PyCharm 
# @Comment :
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

print(F.one_hot(torch.tensor([0, 2]), len(vocab)))
X = torch.arange(10).reshape((2, 5))
X = F.one_hot(X.T, 28)
print('0' * 100)
print(X.type)
print(X[0, :, :])


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_rnn_state(batch_size, num_hidden, device):
    return (torch.zeros((batch_size, num_hidden), device=device))


def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + \
                       torch.mm(H, W_hh) + \
                       b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


class RNNModeScratch:
    '''从零开始实现的循环神经网络模型 '''

    def __init__(self, vocab_size, num_hiddens, device, \
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


num_hiddens = 512
net = RNNModeScratch(len(vocab), num_hiddens, d2l.try_gpu(), \
                     get_params,
                     init_rnn_state, rnn)
state = net.begin_state(X.shape[0],d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
print(Y.shape, len(new_state), new_state[0].shape)
