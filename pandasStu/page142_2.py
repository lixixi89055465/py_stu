# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/25 14:35
@file: page142.py
@desc: 
"""
import pandas as pd
import numpy as np

df = pd.read_csv("./911.csv")
print(df.head(10))
print(df.info())
print(df['title'].head())
temp_list = df['title'].str.split(": ").to_list()
cate_list = [i[0] for i in temp_list]
df['cate']=pd.DataFrame(np.array(cate_list).reshape((df.shape[0],1)),columns=["cate"])
print('2'*100)
print(df.groupby(by='cate').count()['title'])

