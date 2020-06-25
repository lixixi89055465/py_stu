# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/25 14:35
@file: page142.py
@desc: 
"""
import pandas as pd
import numpy as np

df = pd.read_csv("./911.csv")
print(df.head(10))
print(df.info())
print(df['title'].head())
temp_list = df['title'].str.split(": ").to_list()
cate_list = list(set(i[0] for i in temp_list))
print('*'*100)
print(cate_list)

# 构造全为0的数组
zeros_df = pd.DataFrame(np.zeros((df.shape[0], len(cate_list))), columns=cate_list)

# 赋值
for cate in cate_list:
    zeros_df[cate][df['title'].str.contains(cate)] = 1
    # break

sum_set = zeros_df.sum(axis=0)
# print(sum_set)
# print('*1' * 100)
# print(zeros_df.sum(axis=0))
