# -*- coding: utf-8 -*-
# @Time : 2024/1/7 16:54
# @Author : nanji
# @Site : 
# @File : testMake_gaussian_quantiles.py
# @Software: PyCharm 
# @Comment :
from sklearn.datasets import make_gaussian_quantiles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data, target = make_gaussian_quantiles(n_samples=500)
df = pd.DataFrame(data)
df['target'] = target
# df[0].plot(kind='kde', secondary_y=True, label='df[0]')
# df[1].plot(kind='kde', secondary_y=True, label='df[1]')
# plt.legend()
# plt.show()
# 可视化数据
df1 = df[df['target'] == 0]
df2 = df[df['target'] == 2]
df3 = df[df['target'] == 3]

df1.index = range(len(df1))
df2.index = range(len(df2))
df3.index = range(len(df3))

plt.figure(figsize=(5, 5))
plt.scatter(df1[0], df1[1], color='red')
plt.scatter(df2[0], df2[1], color='green')
plt.scatter(df3[0], df3[1], color='orange')
plt.show()
