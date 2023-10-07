# -*- coding: utf-8 -*-
# @Time    : 2023/10/6 上午11:18
# @Author  : nanji
# @Site    : 
# @File    : testLinearKernel.py
# @Software: PyCharm 
# @Comment :
import numpy as np
x=np.random.random(5)#随机向量 x
y=np.random.random(5)# 随机向量 y
print('0'*100)
print(x, y)
K_xy=(x.dot(y))**2
print('1'*100)
print(K_xy)

def fai_func(x):
    # 定义映射，阶次 为2
    vx=[]
    for i in range(len(x)):
        for j in range(len(y)):
            vx.append(x[i]*x[j])
    return np.asarray(vx)

vec_x=fai_func(x)
print('2'*100)
print(vec_x)
vec_y=fai_func(y)
print(vec_y)

print('3'*100)
print(vec_x.dot(vec_y))
