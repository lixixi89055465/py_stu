# -*- coding: utf-8 -*-
# @Time    : 2023/9/17 下午5:17
# @Author  : nanji
# @Site    : 
# @File    : test_logistic_reg2class.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer,load_iris,load_digits
from machinelearn.logistic_regression.logistic_regression_mulclass import LogisticRegression_MulClass
from machinelearn.model_evaluation_selection_02.Performance_metrics import ModelPerformanceMetrics

bc_data = load_digits()  # 加载数据集

X, y = bc_data.data, bc_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

lg_lr = LogisticRegression_MulClass(alpha=0.5, l1_ratio=0.5, batch_size=5, max_epochs=1000, eps=1e-15)
lg_lr.fit(X_train, y_train, X_test, y_test)

print('L1正则化模型参数如下 ：')
theta=lg_lr.get_params()
fn=bc_data.feature_names

for i,w in enumerate(theta[0]):
    print(fn[i],":",w)

print( 'theta:',theta[1])
print('='*100)
y_test_prob=lg_lr.predict_prob(X_test)# 预测概率
y_test_labels=lg_lr.predict(X_test)

pm=ModelPerformanceMetrics(y_test,y_test_prob) # 模型性能度量
print(pm.cal_classification_report())
plt.figure(figsize=(12,8))
plt.subplot(221)
lg_lr.plt_loss_curve(lab='L1',is_show=False)
pr_values=pm.precision_recall_curve()# PR 指标值
plt.subplot(222)
pm.plt_pr_curve(pr_values,is_show=False)
roc_values=pm.roc_metrics_curve()# ROC指标值
plt.subplot(223)
pm.plt_roc_curve(roc_values,is_show=False)# ROC 曲线

plt.subplot(224)
cm=pm.cal_confusion_matrix()
lg_lr.plt_confusion_matrix(cm,label_names=['malignant','benign'],is_show=False )
plt.show()


