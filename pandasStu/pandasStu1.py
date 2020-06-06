# coding: utf-8
# @Time    : 2020/6/6 12:31 PM
# @Author  : lixiang
# @File    : pandasStu1.py
import pandas as pd
import numpy as np
import string

t = pd.Series([1, 2, 32, 12, 3, 4])  # 默认索引
t2 = pd.Series([1, 23, 2, 2, 1], index=list("abcde"))  # 指定索引
# t2 = pd.Series([1, 23, 2, 2, 1], index=list("abcd"))# 指定索引
print(t2)

# 用字典加索引
temp_dict = {"name": "xiaohong", "age": 30, "tel": 10086}
t3 = pd.Series(temp_dict)
print(t3)
# t=pd.Series(np.arange(10), index=list(string.ascii_uppercase[:10]))
print(t3)
print(t3.dtype)
print(t2.dtype)

print(t3["age"])
print("*" * 100)
print(t3[2])
print(t3[[1, 2]])
print("*" * 100)
print(t3[["age", "tel"]])

print(t)
print(t[t > 10])
print(t3.index)
print(t3.values)
print(type(t3))
for i in t3.index:
    print(i)
print("*"*100)
print(type(t3.index))
