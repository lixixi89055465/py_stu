# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/25 10:50
@file: page140.py
@desc: 
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

file_path='./starbucks_store_worldwide.csv'
df=pd.read_csv(file_path)
df=df[df['Country']=='CN']
# 使用matplotlib 呈现出店铺总数排名前十的国家
# 准备数据
data1=df.groupby(by='City').count()['Brand'].sort_values(ascending=False)[:25]
print(data1)
_x=data1.index
_y=data1.values

#画图
plt.figure(figsize=(20,8),dpi=80)
plt.bar(range(len(_x)),_y,width=0.3,color='orange')
plt.xticks(range(len(_x)),_x)
plt.show()

print(df.head(1) )