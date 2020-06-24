# coding: utf-8
# @Time    : 2020/6/6 5:45 PM
# @Author  : lixiang
# @File    : pyMongo.py
from pymongo import MongoClient
import pandas as pd

client=MongoClient()
collection=client["douban"]["tv1"]
data=list(collection.find())


t1=data[0]
t1=pd.Series(t1)
print(t1)
data=[{},{},{}]
df=pd.DataFrame(data)
print(df)