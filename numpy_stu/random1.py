# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/4 18:28
@file: random1.py
@desc: 
"""
import numpy as np

np.random.seed(1)
t = np.random.randint(20, 40, (4, 5))
print(t)
a = np.inf
print(a)
print(np.nan)
print(np.inf < 3)
print(np.nan == np.nan)
print(np.nan != np.nan)
t2 = np.arange(0,24).reshape(4,6)
t2[0,:]=3
t2[:,0]=0
print(t2)
