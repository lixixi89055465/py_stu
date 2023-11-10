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
from machinelearn.linear_model_03.gradient_descent.regularization_linear_regression import \
    RegularizationLinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split

# data = pd.read_csv('../../data/Bias_correction_ucl.csv')
data = pd.read_csv('../../data/Bias_correction_ucl.csv').dropna(axis=0)
X, y = np.asarray(data.iloc[:, 2:-2]), np.asarray(data.iloc[:, -1])
feature_names = data.columns[2:-2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
alpha, batch_size, max_epochs, ratio = 0.1, 200, 5, 0.5

noreg_lr = RegularizationLinearRegression(alpha=alpha, batch_size=batch_size, max_epoch=max_epochs)
noreg_lr.fit(X_train, y_train)
theta = noreg_lr.get_params()
print('无正则化，模型系数如下')
for i, w in enumerate(theta[0][:-1]):
    print(feature_names[i], ":", w)

print('theta0:', theta[1][0])
print('=' * 100)
