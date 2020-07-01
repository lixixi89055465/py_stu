# encoding: utf-8
"""
@author: nanjixiong
@time: 
@file: example06.py
@desc: 
"""
import numpy as np

vector = np.array(['1', '2', '3'])
print(vector.dtype)
vector = vector.astype(float)
print(vector.dtype)
print(vector)

matrix = np.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45],
])
print(matrix.sum(axis=1))

