# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/14 21:38
@file: page113.py
@desc: 
"""
import pandas as pd

df = pd.read_csv("./dogNames2.csv")
print(df.head())
print(df.info())
# 使用dataFrame中排序的方法
df.sort_values(by="Count_AnimalName")
print(df)
t2=pd.DataFrame()
