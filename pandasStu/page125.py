# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/20 9:56
@file: page125.py
@desc: 
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

file_path = '../IMDB-Movie-Data.csv'
df = pd.read_csv(file_path)
# 统计分类的列表
temp_list = df['Genre'].str.split(',').tolist()
genre_list = list(set([j for i in temp_list for j in i]))
print(len(temp_list))

# 构造    全为0 的数组
zeros_df = pd.DataFrame(np.zeros((df.shape[0], len(genre_list))), columns=genre_list)
print(zeros_df)
# 给每个电影出现分类的位置赋值1
for i in range(df.shape[0]):
    # zeros_df.loc[0, ['Sci-fi', 'Musical']] = 1
    zeros_df.loc[i, temp_list[i]] = 1

print('*' * 100)
print(temp_list)
print(df.shape)
print(zeros_df.head(3))
# 统计每个分类的电影的数量和
genre_count = zeros_df.sum(axis=0)
print(genre_count)
# 排序
print(genre_count.sort_values())
_x = genre_count.index
_y = genre_count.values
# 画图
plt.figure(figsize=(20, 8), dpi=80)
plt.bar(range(len(_x)), _y)
plt.xticks(range(len(_x)), _x)
plt.show()
