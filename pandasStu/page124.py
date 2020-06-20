# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/20 8:03
@file: page124.py
@desc: 
"""
import pandas as pd
from matplotlib import pyplot as plt

file_path = '../IMDB-Movie-Data.csv'
df = pd.read_csv(file_path)
print(df.head(1))
print(df.info())
print('*' * 100)
runtime_data = df['Runtime (Minutes)'].values
print(runtime_data)
max_runtime = runtime_data.max()
min_runtime = runtime_data.min()
size=5
# 计算组数
num_bin = (max_runtime - min_runtime) // size
print(num_bin)
# 设置图形的大小
plt.figure(figsize=(20, 8), dpi=80)
plt.hist(runtime_data, num_bin)

plt.xticks(range(min_runtime, max_runtime + size,size))
plt.show()
