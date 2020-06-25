# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/25 11:22
@file: page141.py
@desc: 
"""
import pandas as pd
from matplotlib import pyplot as plt

file_path = './books.csv'
df = pd.read_csv(file_path)
print(df.head())
print(df.info())

print('*' * 100)
d1 = df.dropna(axis=0)
print(d1.info())
# 不同年份书的平均评分情况
# 去除original_publication_year列中的无效数据
data1 = df[pd.notnull(df['original_publication_year'])]
print(data1.info())

grouped=data1['average_rating'].groupby(by=data1['original_publication_year']).count()
print(grouped)
_x=grouped.index
_y=grouped.values
# plt.plot(range(len(_x)),_y)
# plt.plot(range(len(_x)),_x)
# plt.figure(figsize=(20,8),dpi=80)
# 画图
plt.figure(figsize=(20,8),dpi=80)
plt.plot(range(len(_x)),_y)
plt.xticks(list(range(len(_x)))[::100],_x[::100],rotation=90)
plt.show()




# plt.show()