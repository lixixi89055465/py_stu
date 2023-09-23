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
from sklearn.datasets import load_iris,load_digits,load_breast_cancer
from machinelearn.logistic_regression.logistic_regression_mulclass import LogisticRegression_MulClass
from machinelearn.model_evaluation_selection_02.Performance_metrics import ModelPerformanceMetrics
from sklearn.preprocessing import StandardScaler

# iris = load_iris()  # 加载数据集
iris = load_digits()  # 加载数据集
# iris = load_breast_cancer()  # 加载数据集

X, y = iris.data, iris.target
X = StandardScaler().fit_transform(X)  # 标准化
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

lgmc = LogisticRegression_MulClass(alpha=0.5, l1_ratio=0.5,l2_ratio=0.05,en_rou=0.6, batch_size=5, max_epochs=1000,
                                   eps=1e-15, normalize=False)
lgmc.fit(X_train, y_train, X_test, y_test)
plt.figure(figsize=(12, 8))  # 可视化，四个子图
plt.subplot(221)
lgmc.plt_cross_entropy_loss(is_show=False)  # 交叉熵损失下降曲线
y_test_pred = lgmc.predict(X_test)
y_test_prob = lgmc.predict_prob(X_test)  # 预测概率
feature_names = iris.feature_names

for fn, theta in zip(feature_names, lgmc.get_params()[0]):
    print(fn, ':', theta)
print('bias:', lgmc.get_params()[1])  # 偏执项
print('=' * 100)

pm = ModelPerformanceMetrics(y_test, y_test_prob)  # 模型性能度量
print(pm.cal_classification_report())
pr_values = pm.precision_recall_curve()  # PR 指标值
plt.subplot(222)
pm.plt_pr_curve(pr_values, is_show=False)
roc_values = pm.roc_metrics_curve()  # ROC指标值
plt.subplot(223)
pm.plt_roc_curve(roc_values, is_show=False)  # ROC 曲线
cm = pm.cal_confusion_matrix()
plt.subplot(224)
lgmc.plt_confusion_matrix(cm, label_names=None, is_show=False)
plt.show()
