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


def plot_img(df, label):
    count_by_month = df.resample("M").count()['title']
    # 画图
    _x = count_by_month.index
    _y = count_by_month.value
    _x = [i.strftime("%Y%m%d") for i in _x]
    plt.figure(figsize=(20, 8), dpi=80)
    plt.plot(range(len(_x)), _y, label=label)
    plt.xticks(range(len(_x)), _x, rotation=45)
    plt.show()


df = pd.read_csv('./911.csv')
# print(df.info())
# print(df['timeStamp'].head(10))
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
print(df.index)
df.set_index('timeStamp', inplace=True)
# 添加列，表示分类
temp_list = df['title'].str.split(": ").tolist()
cate_list = [i[0] for i in temp_list]
df['cate'] = pd.DataFrame(np.array(cate_list).reshape((df.shape[0], 1)))
# 分组
for group_name, group_data in df.groupby(by='cate'):
    # 对不同的分类都进行绘图
    plot_img(group_data, group_name)

plt.xticks(range(len(_x)),_x,rotation=45)
plt.show()
