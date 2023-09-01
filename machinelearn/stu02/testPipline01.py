# -*- coding: utf-8 -*-
# @Time    : 2023/9/1 9:52
# @Author  : nanji
# @Site    : 
# @File    : testPipline.py
# @Software: PyCharm 
# @Comment :

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
print('0'*100)
wdbc=pd.read_csv('D:\workspace\py_stu\machinelearn\breast+cancer+wisconsin+diagnostic\wdbc.data')
print(wdbc.info())
pip_lr=make_pipeline(StandardScaler(),
                     PCA(n_components=6),
                     LogisticRegression() )
# 划分数据集为训练集和测试集，比例8:2
X_train,X_test,y_train,y_test=train_test_split(X,y,)

