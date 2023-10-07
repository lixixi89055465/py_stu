# -*- coding: utf-8 -*-
# @Time    : 2023/10/7 17:23
# @Author  : nanji
# @Site    : 
# @File    : testROC.py
# @Software: PyCharm 
# @Comment :  3. 性能度量——ROC与AUC+二分类
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import random

breatcaner = pd.read_csv('../data/breast-cancer.csv', header=None).iloc[:, 1:]
X = StandardScaler().fit_transform(breatcaner.iloc[1:, 2:])
y = breatcaner.iloc[1:, 0]
print(X.shape)
print(y.shape)
n_samples, n_features = X.shape
random_state = np.random.RandomState(0)
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]  # 添加噪声
y = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
y_score = dict()
from sklearn.svm import SVC

svm_linear = SVC(kernel='linear', probability=True, random_state=0)
# 通过decision_function(0计算得到的y_score 的值，用在roc_curve()函数中
svm_fit = svm_linear.fit(X_train, y_train)
y_score['svm_linear'] = svm_linear.decision_function(X_test)
from sklearn.linear_model import LogisticRegression

lg_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)  # 逻辑回归
y_score['lg'] = lg_model.decision_function(X_test)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda_model = LinearDiscriminantAnalysis().fit(X_train, y_train)
y_score['lda'] = lda_model.decision_function(X_test)

from sklearn.ensemble import AdaBoostClassifier

abc_model=AdaBoostClassifier().fit(X_train,y_train)
y_score['abc']=abc_model.decision_function(X_test)

from sklearn.metrics import roc_curve
fpr,tpr,threshold,ks_max,best_thr=dict(),dict(),dict(),dict(),dict()
for key in y_score.keys():
    # 计算真正率和假正 率
    fpr[key],tpr[key],threshold[key]=roc_curve(y_test,y_score[key])
    # 计算ks和最佳阈值
    KS_max=tpr[key]-fpr[key]# 差值向量
    ind=np.argmax(KS_max)
    best_thr[key]=threshold[key][ind]
    print(ind)

import matplotlib.pyplot  as plt
for i,key in enumerate(y_score.keys()):
    plt.plot(fpr[key],tpr[key],lw=2)
plt.show()




