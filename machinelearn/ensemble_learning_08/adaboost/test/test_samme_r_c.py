# -*- coding: utf-8 -*-
# @Time    : 2023/10/15 下午4:56
# @Author  : nanji
# @Site    : 
# @File    : test_adaboost_c.py
# @Software: PyCharm 
# @Comment :8.2.3
# from machinelearn.decision_tree_04.decision_tree_C \
#     import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from machinelearn.ensemble_learning_08.adaboost.utils.plt_decision_function import plot_decision_function

from sklearn.datasets import make_classification, make_blobs
from machinelearn.ensemble_learning_08.adaboost.adaboost_discrete_c \
    import AdaBoostClassifier
from sklearn.metrics import classification_report
from machinelearn.linear_model_03.logistic_regression.logistic_regression2class import LogisticRegression
from machinelearn.svm_06.svm_smo_classifier import SVMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from machinelearn.ensemble_learning_08.adaboost.samme_r_multi_classifier import SAMMERClassifier
from sklearn.datasets import load_iris, load_digits
from sklearn.decomposition import PCA

# X, y = make_blobs(n_samples=1000, n_features=10, centers=5, cluster_std=[1.5, 2, 0.9, 3, 2.8], random_state=0)
# X = StandardScaler().fit_transform(X)
# base_em = DecisionTreeClassifier(max_depth=4, is_feature_all_R=True, max_bins=10)
digits = load_digits()
X, y = digits.data, digits.target
X = PCA(n_components=10).fit_transform(X)
base_em = DecisionTreeClassifier(max_depth=4)
acc_scores = []  # 存储每次交叉验证的均分

# 用 10折交叉验证评估不同基学习器个数T 下的分类正确率
for n in range(1, 201):
    scores = []  # 一次交叉验证的acc均值
    k_fold = KFold(n_splits=10)
    for idx_train, idx_test in k_fold.split(X, y):
        classifier = SAMMERClassifier(base_estimator=base_em, n_estimators=n)
        classifier.fit(X[idx_train, :], y[idx_train])
        y_test_pred = classifier.predict(X[idx_test, :])
        scores.append(accuracy_score(y[idx_test], y_test_pred))
    acc_scores.append(np.mean(scores))
    print(n, ':', acc_scores[-1])

plt.figure(figsize=(7, 5))
plt.plot(range(1, 21), acc_scores, 'ko-', lw=1)
plt.xlabel('Number of Estimcations', fontdict={'fontsize': 12})
plt.ylabel('Accuracy Score', fontdict={'fontsize': 12})
plt.title('Cross Validation Scores of Different Number of Base Learners', fontdict={'fontsize': 12})
plt.grid(ls=':')
plt.show()
