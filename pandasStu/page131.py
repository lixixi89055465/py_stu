# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/25 8:50
@file: page131.py
@desc: 
"""
import numpy as np
import pandas as pd

# df1=pd.DataFrame([[100,1,1,1],[1,1,1,1]],index=['A','B'],columns=['a','b', 'c','d'],dtype=np.float)
df1=pd.DataFrame([[100,1,1,1],[1,1,1,1]],index=list('AB'),columns=list('abcd'),dtype=np.float)
print(df1)

print(df1.index)
df1.index=['a','b']
print(df1)
print(df1.reindex(['a', 'f']))
# 把当前某一列作为索引
print(df1.set_index('a'))
print(df1.set_index('a', drop=False))
print(df1['d'].unique())
print(df1['a'].unique())
print('*'*100)
print(df1.set_index("b").index)
print('*'*100)
print(df1.set_index('a', 'b'))
print(df1.set_index(['a', 'b', 'c'], drop=False))
a=pd.DataFrame({'a':range(7),'b':range(7,0,-1),'c':['one','one','one','two','two','two','two'],'d':list('hjklmno')})
print(a)
# b=a.set_index('a','b')
# print('*'*100)
# print(b)
b=a.set_index(['c','d'])
print('*'*100)
c=b['a']
print(c)
print(type(c))
print(c['one'])
print(c['two'])
d=a.set_index(['d','c'])['a']
print(d)
print(d.index)
# 从内层索引取值，
print(d.swaplevel()['one'])
print(b)
# print(b['a'])

# print(b.loc['one']['a'])

print(b.loc['one'].loc['h'])