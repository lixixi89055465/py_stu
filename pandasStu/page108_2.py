# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/14 20:58
@file: page108_2.py
@desc: 
"""
import pandas as pd

d2 = [
    {"name": "xiaohong", "age": 32, "tel": 10010},
    {"name": "xiaogang", "tel": 10000},
    {"name": "xiaowang", "age": 32}
]
t2 = pd.DataFrame(d2)
print(t2)
print(t2.index)
print(t2.columns)
print(t2.shape)
print("*" * 100)
print(t2.dtypes)
print("*" * 100)
print(t2.values)
print(t2.keys)
print("*" * 40)
print(t2.dtypes)
print(t2.ndim)

print(t2.head())
print("*" * 40)
print(t2.tail(3))
print("*" * 40)
print(t2.info())
print("*" * 40)

print(t2.describe())
