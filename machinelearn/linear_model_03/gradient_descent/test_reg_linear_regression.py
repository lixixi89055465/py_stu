# -*- coding: utf-8 -*-
# @Time    : 2023/9/15 下午11:34
# @Author  : nanji
# @Site    : 
# @File    : test_reg_linear_regression.py
# @Software: PyCharm 
# @Comment :

import numpy as np
import matplotlib.pyplot as plt
from machinelearn.model_evaluation_selection_02.Polynomial_feature import PolynomialFeatureData
from machinelearn.linear_model_03.gradient_descent.regularization_linear_regression import RegularizationLinearRegression

def objective_fun(x):
    '''
    目标函数
    :param x:
    :return:
    '''
    return 0.5*x**2+x+2

np.random.seed(42)
n=30 # 采样数据的样本量
raw_x=np.sort(6*np.random.randn(n-1)-3,axis=0)# [-3,3]区间
raw_y=objective_fun(raw_x)# 二维数组
X_test_raw=np.linspace(-3,3,150)# 测试数据

feature_obj=PolynomialFeatureData(raw_x,degree=13,with_bias=False)
X_train=feature_obj.fit_transform()# 特征函数的构造

x_test_raw=np.linspace(-3,3,100)
feature_obj=PolynomialFeatureData(x_test_raw,degree=13,with_bias=False)

X_test=feature_obj.fit_transform()# 数据的构造
y_test=objective_fun(x_test_raw)#  测试样本真值




reg_ratio=[0.1,0.5,1,2,3,5]# 正则化系数
alpha,batch_size,max_epochs=0.1,10,300
plt.figure(figsize=(15,8))

for i,ratio in enumerate(reg_ratio):
    plt.subplot(231+i)
    # 不采用正则化
    reg_lr=RegularizationLinearRegression(solver='form',alpha=alpha,batch_size=batch_size,max_epoch=max_epochs)
    reg_lr.fit(X_sample,raw_y)






