# -*- coding: utf-8 -*-
# @Time : 2024/1/28 16:55
# @Author : nanji
# @Site : 
# @File : test_CartPrune.py
# @Software: PyCharm 
# @Comment :CART 后剪枝 测试案例
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from machinelearn.decision_tree_04.decision_tree_C import DecisionTreeClassifier

data, target = make_classification(n_samples=100, \
								   n_features=2, \
								   n_classes=2, \
								   n_informative=1, \
								   n_redundant=0, \
								   n_repeated=0, \
								   n_clusters_per_class=1,
								   class_sep=0.9, random_state=21)
print(data.shape)
print(target.shape)
tree = DecisionTreeClassifier(is_feature_all_R=True)
tree.fit(data, target)
y_test_pred = tree.predict(data)
print(classification_report(target, y_test_pred))

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

acc = accuracy_score(target, y_test_pred)
plt.figure(figsize=(14, 10))
plt.subplot(221)
from machinelearn.decision_tree_04.utils.plt_decision_function import plot_decision_function

plot_decision_function(data, target, tree, acc, is_show=False, \
					   title_info='By CART UnPrune')
alpha = [1, 3, 5]
for i in range(3):
	tree.prune(alpha, alpha[i])
	y_test_pred = tree.predict(data)
	print(classification_report(target, y_test_pred))
	acc = accuracy_score(target, y_test_pred)
	plt.subplot(222 + i)
	plot_decision_function(data, target, tree, acc, \
						   is_show=False, \
						   title_info='By Cart PostPrune a= %.1f' % alpha[i])
plt.show()
