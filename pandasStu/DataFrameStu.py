# coding: utf-8
# @Time    : 2020/6/6 4:55 PM
# @Author  : lixiang
# @File    : DataFrameStu.py
import pandas as pd
import numpy as np

d1 = pd.DataFrame(np.arange(12).reshape(3, 4))
print(d1)
p1 = pd.DataFrame(np.arange(12).reshape(3, 4), index=list("abc"), columns=list("WXYZ"))
print(p1)
p1 = pd.DataFrame(np.arange(12).reshape(3, 4), index=list("abc"), columns=list("WXYZ"))
print(p1)
print("*" * 100)
d1 = {"name": ["xiaoming", "xiaogang"], "age": [20, 32], "tel": [10086, 10010]}
t1 = pd.DataFrame(d1)
print("*" * 100)
print(type(t1))
d2 = [
    {"name": "xiaohong", "age": 32, "tel": 10010},
    {"name": "xiaogang", "tel": 10000},
    {"name": "xiaowang", "age": 32}
]
t2=pd.DataFrame(d2)
print(t2)
print(t2.mean())
print(t2.fillna(t2.mean()))
t3=t2.dropna(axis=0,how='any')
print(t3)
t3=t2.dropna(axis=0,how='all')
print(t3)
t2.dropna(axis=0,how='any',inplace=True)
print(t2)
