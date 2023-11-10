# -*- coding: utf-8 -*-
# @Time    : 2023/9/30 19:45
# @Author  : nanji
# @Site    : 
# @File    : test_decision_tree_C_2.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from machinelearn.decision_tree_04.decision_tree_C import DecisionTreeClassifier
from machinelearn.decision_tree_04.utils.plt_decision_function import plot_decision_function

# 生成数据
data, target = make_classification(n_samples=200, n_features=2, n_classes=2, n_informative=1,
                                   n_redundant=0, n_clusters_per_class=1,
                                   class_sep=0.8, random_state=21)
cart_tree = DecisionTreeClassifier(is_feature_all_R=True)
cart_tree.fit(data, target)
y_test_pred = cart_tree.predict(data)
print(classification_report(target, y_test_pred))
plt.figure(figsize=(14, 10))
plt.subplot(221)
from sklearn.metrics import accuracy_score

acc = accuracy_score(target, y_test_pred)
plot_decision_function(data, target, cart_tree, acc=acc, is_show=False, title_info="By CART unPrune")

# 剪枝处理
alpha = [1, 3, 5]
# alpha = [1, 2, 3]
for i in range(3):
    cart_tree.prune(alpha=alpha[i])
    y_test_pred = cart_tree.predict(data)
    acc = accuracy_score(target, y_test_pred)
    plt.subplot(222 + i)
    plot_decision_function(data, target, cart_tree,
                           acc=acc, is_show=False,
                           title_info='By CART Prune a = %.1f' % alpha[i])
plt.show()
