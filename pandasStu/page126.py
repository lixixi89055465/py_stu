# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/20 16:50
@file: page126.py
@desc: 
"""
import pandas as pd
import numpy as np

df1 = pd.DataFrame(np.ones((2, 4)), index=list('AB'), columns=list('abcd'))
print(df1)
df2 = pd.DataFrame(np.zeros((3, 3)), index=['A', 'B', 'C'], columns=list('xyz'))
print(df2)
print(df1.join(df2))
print(df2.join(df1))
df3 = pd.DataFrame(np.zeros((3, 3,)), columns=list('fax'))
print('*' * 100)
print(df1.merge(df3, on='a'))
print(df3)
df3.loc[1, 'a'] = 1
print(df3)
df3 = pd.DataFrame(np.arange(9).reshape((3, 3)), columns=list("fax"))
print(df3)
print(df1.merge(df3))
print('*' * 100)
df1.loc['A', 'a'] = 100
print(df1)
print(df1.merge(df3, on="a"))
print('*' * 100)
print(df1.merge(df3, on='a', how='inner'))
print(df1.merge(df3, on='a', how='outer'))
print('right' + '*' * 100)
print(df1.merge(df3, on='a', how='right'))
print('left' + '*' * 100)
print(df1.merge(df3, on='a', how='left'))
