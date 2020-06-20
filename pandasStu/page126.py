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
