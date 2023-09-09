# -*- coding: utf-8 -*-
# @Time    : 2023/9/9 18:11
# @Author  : nanji
# @Site    : 
# @File    : test_linear_regression_cfsol.py
# @Software: PyCharm 
# @Comment :
import numpy as np
from sklearn.datasets import load_boston
from machinelearn.linear_model_03.closed_form_sol.LinearRegression_CFSol import LinearRegressionClosedFormSol
from sklearn.model_selection import train_test_split

boston=load_boston()#加载数据
X,y=boston.data,boston.target#样本数据和目标值
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0,shuffle=True,stratify=y)
lr_cfs=LinearRegressionClosedFormSol()#使用默认设置偏置项和标准化
lr_cfs.fit(X_train,y_train)
theta=lr_cfs.get_params()

print(theta)


