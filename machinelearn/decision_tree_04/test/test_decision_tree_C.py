# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/9/29 0:55
# @Author  : nanji
# @File    : test_decision_tree_C.py
# @Description :
import matplotlib.pyplot as plt
import numpy as np

from machinelearn.decision_tree_04.decision_tree_C import DecisionTreeClassifier
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split

# data = pd.read_csv('../../data/watermelon.csv').iloc[:, 1:]
# X, y = data.iloc[:, :-1], data.iloc[:, -1]
# iris = load_iris()
# X, y = iris.data, iris.target

# bc_data = load_breast_cancer()
# X, y = bc_data.data, bc_data.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
nursery = pd.read_csv('../../data/nursery.csv')
X, y = np.asarray(nursery.iloc[:, :-1]), np.asarray(nursery.iloc[:, -1])
from sklearn.preprocessing import LabelEncoder
y=LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                    stratify=y)

# dtc = DecisionTreeClassifier(dbw_feature_idx=[6, 7], max_bins=8)
# dtc.fit(X,y)
# dtc = DecisionTreeClassifier(is_feature_all_R=True, max_bins=5)
# dtc.fit(X_train, y_train)
# y_pred_labels = dtc.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score

depth = np.linspace(2, 10, 9, dtype=np.int)
accuracy = []
for d in depth:
    dtc = DecisionTreeClassifier(is_feature_all_R=False, max_depth=d)
    dtc.fit(X_train, y_train)
    y_pred_labels = dtc.predict(X_test)
    accuracy.append(accuracy_score(y_test, y_pred_labels))
plt.figure(figsize=(7, 5))
plt.plot(depth, accuracy, 'ko-', lw=1)
plt.show()

# print('0' * 100)
# print(classification_report(depth, y_pred_labels))
