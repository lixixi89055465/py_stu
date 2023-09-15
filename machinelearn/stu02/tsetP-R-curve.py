# -*- coding: utf-8 -*-
# @Time    : 2023/9/15 16:55
# @Author  : nanji
# @Site    : 
# @File    : tsetP-R-curve.py
# @Software: PyCharm 
# @Comment : 3. 性能度量——P-R曲线代码示例，sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,label_binarize
from sklearn import datasets
from sklearn.model_selection import train_test_split


def data_preproce():
    #加载数据，数据预处理
    digits=datasets.load_digits()
    X,y=digits.data,digits.target
    n_samples,n_features=X.shape
    random_state=np.random.RandomState(0)
    print(X.shape)
    X=np.c_[X,random_state.randn(n_samples,10*n_features )]# 添加噪声特征
    X=StandardScaler().fit_transform(X)#标准化
    y=label_binarize(y,classes=np.unique(y))# one-hot
    # 划分数据集
    X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=0,shuffle=True,stratify=y )
    return X_train,X_test,y_train,y_test

def model_train(model):
    pass


def micro_PR(y_test, y_score):
    pass


def plt_PR_curve(precision, recall, average_precision, label):
    pass


X_train, X_test, y_train, y_test = data_preproce()
