# -*- coding: utf-8 -*-
# @Time    : 2023/9/30 19:45
# @Author  : nanji
# @Site    : 
# @File    : test_decision_tree_C_2.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report,accuracy_score
from sklearn.datasets import make_classification
from machinelearn.decision_tree_04.decision_tree_C import DecisionTreeClassifier
from machinelearn.decision_tree_04.utils.plt_decision_function import plot_decision_function
from sklearn.model_selection import StratifiedKFold
import copy

# 生成数据
bc_data = load_breast_cancer()
X, y = bc_data.data, bc_data.target

alphaArr = np.linspace(0, 10, 3)  # 存储每个alpha阈值下的交叉验证均分
accuracy_scores = []  # 存储每个alpha阈值下的交叉验证均分
cart = DecisionTreeClassifier(criterion='cart', is_feature_all_R=True, max_bins=10)
for alpha in alphaArr:
    scores = []
    k_fold = StratifiedKFold(n_splits=10).split(X, y)
    for train_idx, test_idx in k_fold:
        tree = copy.deepcopy(cart)
        tree.fit(X[train_idx], y[train_idx])
        tree.prune(alpha=alpha)
        y_test_pred = tree.predict(X[test_idx])
        scores.append(accuracy_score(y[test_idx], y_test_pred))
        del tree
    print(alpha, ':', np.mean(scores))
    accuracy_scores.append(np.mean(scores))
plt.figure(figsize=(7, 5))
plt.plot(alphaArr, accuracy_scores, 'ko-', lw=1)
plt.grid(ls=':')
plt.xlabel('Alpha', fontdict={'fontsize': 12})
plt.ylabel('Accuracy Scores', fontdict={'fontsize': 12})
plt.title('Cross Validation scores under different Pruning Alpha', fontdict={'fontsize': 14})
plt.show()
