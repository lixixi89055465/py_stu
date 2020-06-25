# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/20 18:18
@file: page130.py
@desc: 
"""
import pandas as pd
import numpy as np

file_path = './starbucks_store_worldwide.csv'
df = pd.read_csv(file_path)
# print(df.head(1))
# print(df.info())

# grouped=df.groupby(by="Country")
# print(grouped)

# DataFrameGroupBy
# 可以进行遍历
# for i,j in grouped:
#     print(i)
#     print("-"*100)
#     print(j,type(j))
#     print("*"*100)


# 统计中国每个省店铺的数量
# china_data=df[df['Country']=='CN']
# print(china_data)
# grouped=china_data.groupby(by="State/Province").count()['Brand']
# print(grouped)
# print("*"*100)
# grouped=df.groupby(by=[df["Country"],df["State/Province"]])
# print(grouped.head())
# print("*"*100)
# 数据按照多个条件尽心分组
# print(df['Brand'].head(10))
# print("*1"*100)

# print(df['Brand'].groupby(by=[df['Country'], df['State/Province']]).count())
# grouped = df[['Brand']].groupby(by=[df['Country'], df['State/Province']]).count()
# df.to_csv('a.csv', sep='\t', index=False)
df['Brand'].to_csv('b.csv', sep='\t', index=False)
grouped1 = df[['Brand']].groupby(by=[df['Country'], df['State/Province']]).count()
# print(grouped1)
# print(df[['Brand']])
grouped2 = df.groupby(by=[df['Country'], df['State/Province']])[['Brand']].count()
# print(grouped2)
grouped3 = df.groupby(by=[df['Country'], df['State/Province']]).count()['Brand']
# print(grouped3)
print(type(grouped1))
print(grouped1.index)
