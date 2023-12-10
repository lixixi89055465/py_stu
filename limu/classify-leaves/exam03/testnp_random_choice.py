# -*- coding: utf-8 -*-
# @Time    : 2023/12/8 下午9:16
# @Author  : nanji
# @Site    : 
# @File    : testnp_random_choice.py
# @Software: PyCharm 
# @Comment :

import numpy as np

# 从[0, 5)中随机输出一个随机数
print(np.random.choice(5))
print('0' * 100)
# 在[0, 5)内输出五个数字并组成一维数组（ndarray）
print(np.random.choice(5, 3))

print(np.random.randint(0, 5, 3))
print('1' * 100)
L = [1, 2, 3, 4, 5]  # list列表
T = (2, 4, 6, 2)  # tuple元组
A = np.array([4, 2, 1])  # numpy,array数组,必须是一维的
A0 = np.arange(10).reshape(2, 5)  # 二维数组会报错

print('2' * 100)
print(np.random.choice(L, 5,replace=False))
print(np.random.choice(L, 5,replace=False))
print(np.random.choice(L, 5,replace=True))
# np.random.choice(A0, 5)  # 如果是二维数组，会报错
