# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/20 6:43
@file: page122.py
@desc: 
"""
import pandas as pd
import numpy as np

file_path = '../IMDB-Movie-Data.csv'
df = pd.read_csv(file_path)
print(len(df['Metascore']))
print(df['Metascore'].mean())

print(df.info())
print(df.head(1))
# 获取平均评分
print(df['Rating'].mean())
# 导演的人数
print(set(df['Director'].tolist()))
print(df['Director'].unique())
print(len(df['Director'].unique()))
print(df['Actors'].str.split(', ').tolist())
actorList = [j for i in df['Actors'].str.split(', ').tolist() for j in i]
print(np.array(actorList).flatten())
