# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/25 16:43
@file: page143test.py
@desc: 
"""
import numpy as np
import pandas as pd

print(pd.date_range(start='20171230', end='20180131', freq='10D'))
print(type(pd))
print(pd)
print('1'*100)
print(pd.date_range(start='20171230', periods=5, freq='D'))
print(pd.date_range(start='20171230', periods=10, freq='H'))
print(pd.date_range(start='20171230', periods=10, freq='H'))
print(pd.date_range(start='20171230', periods=10, freq='MS'))
print(pd.date_range(start='2017/12/30', periods=10, freq='MS'))
print(pd.date_range(start='2017-12-30', periods=10, freq='MS'))
