# coding: utf-8
# @Time    : 2020/6/6 4:55 PM
# @Author  : lixiang
# @File    : DataFrameStu.py
import pandas as pd
import numpy as np

d1 = pd.DataFrame(np.arange(12).reshape(3, 4))
print(d1)
d2=pd.DataFrame(np.arange(12).reshape(3,4),index=list("abc"),columns=list("WXYZ"))
print(d2)

d1={"name":["xiaoming","xiaogang"],"age":[20,32],"tel":[10086,10010]}
t1=pd.DataFrame(d1)
print(t1)
print(type(t1))
d2=[
    {"name":"xiaohong","age":32,"tel":10010},
    {"name":"xiaowang","tel":10000},
    {"name":"xiaoli","age":22}
    ]
t2=pd.DataFrame(d2)
print(t2)
print(t2.index)
print(t2.columns)
print(t2.values)
print(t2.shape)
print(t2.dtypes)
print(type(t2))
print("-"*100)
print(t2.head())
print(t2.head(2))
print(t2.tail(3))
print("*" * 100)
print(t2.info())
print("*" * 100)
print(t2.describe())

# dataFrame 中排序的方法
df=t2.sort_values(by="name",ascending=False)
print(df.head(2))
t3=pd.DataFrame(np.arange(12).reshape(3,4),index=list("abc"),columns=list("WXYZ"))
print(t3)
print(type(t3.loc["a","Z"]))
print(t3.loc["a"])
print(t3.loc["a",:])
print(t3.loc[:,"Y"])
print(t3.loc[["a", "c"]])
print("*"*100)
print(t3)
print(t3.loc[["a", "c"], :])
print(t3.loc[:, ["W", "Z"]])

print(t3.loc[["a", "b"], ["W", "Z"]])

print("*"*100)
print(t3.iloc[:,[2,1]])
print(t3.iloc[[0,2],[2,1]])
t3.iloc[1:,:2]=np.nan
print(t3)
