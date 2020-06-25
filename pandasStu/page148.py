# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/25 18:00
@file: page148.py
@desc: 
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('./911.csv')
# print(df.info())
# print(df['timeStamp'].head(10))
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
print(df.index)
df.set_index('timeStamp', inplace=True)
# print(df.info())
print('1' * 100)
print(df.index)

# 统计出911数据中不同月份电话次数的
count_by_month = df.resample('M').count()['title']
print('2' * 100)
# print(count_by_month.head())
# 画图
_x = count_by_month.index
_y = count_by_month.values
plt.figure(figsize=(20, 8), dpi=80)
plt.plot(range(len(_x)) , _y)
plt.xticks(range(len(_x)), _x, rotation=45)
plt.show()
