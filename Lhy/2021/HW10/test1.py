# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 08:59:50 2020

@author: 周文青

利用torch.autograd计算单变量标量函数y=x^3+sin(x)在x分别为1，pi和5时的一阶导数和二
阶导数

"""
import torch as tc
import numpy as np

#%% 方法1：采用torch.autograd.grad
x  = tc.tensor([1, np.pi, 5],requires_grad=True)
y = x**3 +  tc.sin(x)
dy = 3*x**2 + tc.cos(x)
d2y = 6*x - tc.sin(x)

dydx = tc.autograd.grad(y, x,
                        grad_outputs=tc.ones(x.shape), #注意这里需要人为指定
                        create_graph=True,
                        retain_graph=True) # 为计算二阶导保持计算图
print(dydx) # 注意输出是一个tuple，取第一个元素
# (tensor([ 3.5403, 28.6088, 75.2837], grad_fn=<AddBackward0>),)
print(dy)
# tensor([ 3.5403, 28.6088, 75.2837], grad_fn=<AddBackward0>)

d2ydx2 = tc.autograd.grad(dydx[0],x,
                          grad_outputs=tc.ones(x.shape),
                          create_graph=False) # 默认会自动销毁计算图
print(d2ydx2)
# (tensor([ 5.1585, 18.8496, 30.9589]),)
print(d2y)
# tensor([ 5.1585, 18.8496, 30.9589], grad_fn=<SubBackward0>)

#%% 方法2：采用torch.autograd.backword
x  = tc.tensor([1, np.pi, 5],requires_grad=True)
y = x**3 +  tc.sin(x)
dy = 3*x**2 + tc.cos(x)
d2y = 6*x - tc.sin(x)

print('1'*100)
tc.autograd.backward(y, grad_tensors=tc.ones(x.shape),
           create_graph=True, retain_graph=False)
print(x.grad) #一阶导
# tensor([ 3.5403, 28.6088, 75.2837], grad_fn=<CopyBackwards>)
tc.autograd.backward(x.grad, grad_tensors=tc.ones(x.shape),
           create_graph=False, retain_graph=False)

#采用backword的方法并且在求一阶导的时候设置了create_graph时，该结果是两次梯度的累加结果
print(x.grad)
# tensor([  8.6988,  47.4584, 106.2426], grad_fn=<CopyBackwards>)
