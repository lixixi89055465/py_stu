# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/10/19 21:20
# @Author  : nanji
# @File    : test_adaboost_regression.py
# @Description :

import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
from machinelearn.decision_tree_04.decision_tree_R import DecisionTreeRegression
from machinelearn.ensemble_learning_08.adaboost.adaboost_regression import AdaBoostRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
boston = load_boston()
X, y = boston.data, boston.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
base_ht = DecisionTreeRegression(max_bins=50, max_depth=5)
# abr = AdaBoostRegression(base_estimator=base_ht, n_estimators=3, comb_strategy='weight_mean')
plt.figure(figsize=(14,15))
def train_plot(cs,loss,i):
    abr=AdaBoostRegression(base_estimator=base_ht,n_estimators=30,\
                           comb_strategy=cs,loss=loss)
    abr.fit(X_train,y_train)
    y_hat=abr.predict(X_test)
    plt.subplot(231+i)
    idx=np.argsort(y_test)# 对真值做排序
    plt.plot(y_test[idx],'k-',lw=1.5,label='Test true')
    plt.plot(y_hat[idx],'r-',lw=1,label='Predict')
    plt.legend(frameon=False)
    plt.title('%s,%s,R2=%.5f,MSE=%.5f'% \
              (cs,loss,r2_score(y_test,y_hat),((y_test-y_hat)**2).mean()))
    plt.xlabel("Test Samples Serial Number",fontdict={'fontsize':12})
    plt.ylabel('True vs Predict',fontdict={'fontsize':12})
    plt.grid(ls=':')
    print(cs,loss)
loss_func=['linear','square','exp']
comb_strategy=['weight_mean','weight_median']
i=0
for loss in loss_func:
    for cs in comb_strategy:
        train_plot(cs,loss,i)
        i+=1