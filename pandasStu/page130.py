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
print(df.head(1))
print(df.info())
