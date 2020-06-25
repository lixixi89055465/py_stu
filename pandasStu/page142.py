# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/25 14:35
@file: page142.py
@desc: 
"""
import pandas as pd

df = pd.read_csv("./911.csv")
print(df.head(10))
print(df.info())
print('*'*100)
print(df['title'].head())
