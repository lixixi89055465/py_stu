# -*- coding: utf-8 -*-
# @Time : 2024/1/16 11:07
# @Author : nanji
# @Site : 
# @File : 10_2.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l

n_train = 50
x_train, _ = torch.sort(torch.rand(n_train) * 5)


def f(x):
	return 2 * torch.sin(x) + x ** 0.8


y_train = f(x_train) + torch.normal(0, 0.5, (n_train,))
x_test = torch.arange(0, 5, 0.1)
y_truth = f(x_test)
n_test = len(x_test)
print('0' * 100)
print(n_test)

def plot_kernel_reg(y_hat):
	d2l.plot(x_test,[y_truth,y_hat],'x')
