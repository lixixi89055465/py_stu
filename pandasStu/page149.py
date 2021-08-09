# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/26 8:19
@file: page149.py
@desc: 
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

pd.set_option('display.width', 1000)  # 加了这一行那表格的一行就不会分段出现了
pd.set_option('display.max_columns', 20)  # 显示所有行
pd.set_option('display.max_rows', None)  # 显示所有列

file_path = './PM2.5/BeijingPM20100101_20151231.csv'
df = pd.read_csv(file_path)
# print(df.head())
print(df.info())
period = pd.PeriodIndex(year=df['year'], month=df['month'], day=df['day'], hour=df['hour'], freq='H')
# print(period)
# print(type(period))
df['datetime'] = period
df.set_index(['datetime'], inplace=True)
# 进行降采样
df = df.resample("7D").mean()
print(df.shape)
# 处理缺失数据，删除缺失数据
# images = df['PM_US Post'].dropna();
# data_china=df['PM_Dongsi'].dropna()
data = df['PM_US Post'];
data_china=df['PM_Dongsi']
# 画图
_x = data.index
_x = [i.strftime("%Y%m%d") for i in _x]
_x_china=[i.strftime("%Y%m%d") for i in data_china.index]
_y = data.values
_y_china=data_china.values
print(len(_y))
print(len(_y_china))
plt.figure(figsize=(20, 8), dpi=80)
plt.plot(range(len(_x)), _y,label='US_POST')
plt.plot(range(len(_x_china)), _y_china,label='CN_POST')

plt.xticks(range(0, len(_x), 10), list(_x)[::10], rotation=45)
plt.legend(loc='best')

plt.show()
