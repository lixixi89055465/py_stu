# -*- coding: utf-8 -*-
# @Time    : 2023/10/15 下午4:56
# @Author  : nanji
# @Site    : 
# @File    : test_adaboost_c.py
# @Software: PyCharm 
# @Comment :8.2.3
from machinelearn.decision_tree_04.decision_tree_C \
    import DecisionTreeClassifier
from machinelearn.ensemble_learning_08.adaboost.utils.plt_decision_function import plot_decision_function

from sklearn.datasets import make_classification
from machinelearn.ensemble_learning_08.adaboost.adaboost_discrete_c \
    import AdaBoostClassifier
from sklearn.metrics import classification_report
from machinelearn.linear_model_03.logistic_regression.logistic_regression2class import LogisticRegression
from machinelearn.svm_06.svm_smo_classifier import SVMClassifier

X, y = make_classification(n_samples=300, n_features=2, \
                           n_informative=1, n_redundant=0, \
                           n_classes=2, n_clusters_per_class=1, \
                           class_sep=1, random_state=42)
# 同质，同种类型的集学习器
base_tree = DecisionTreeClassifier(max_depth=5, is_feature_all_R=True, \
                                   max_bins=20)
ada_bc = AdaBoostClassifier(base_estimator=base_tree, n_estimators=10, learning_rate=1.0)
ada_bc.fit(X, y)  # adabooo=st 训练
print('基学习器的权重系数:\n', ada_bc.estimator_weights)
y_pred = ada_bc.predict(X)  # 预测类别
print(classification_report(y, y_pred))
plot_decision_function(X, y, ada_bc)

# 异质，不同类型的基学习器
log_reg = LogisticRegression(batch_size=20, max_epochs=5)

cart = DecisionTreeClassifier(max_depth=4, is_feature_all_R=True)
svm = SVMClassifier(C=5.0, max_epochs=20)
ada_bc2 = AdaBoostClassifier(base_estimator=[log_reg, cart, svm], learning_rate=1.0)
ada_bc2.fit(X, y)  # adaboost 训练
print('异质基学习器的权重系数 ：', ada_bc2.estimator_weights)
y_pred = ada_bc2.predict(X)  # 预测类别
print(classification_report(y, y_pred))
plot_decision_function(X, y, ada_bc2)
