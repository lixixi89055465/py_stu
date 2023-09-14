# -*- coding: utf-8 -*-
# @Time    : 2023/9/1 15:35
# @Author  : nanji
# @Site    : 
# @File    : testGridSearchCV.py
# @Software: PyCharm 
# @Comment : 2. 评估方法——嵌套交叉验证(选择算法)
from sklearn.model_selection import GridSearchCV, train_test_split,cross_val_score
from sklearn.svm import SVC  # 支持向量机
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

pipe_svc = make_pipeline(StandardScaler(), PCA(n_components=4), SVC())
wdbc = pd.read_csv('../breast+cancer+wisconsin+diagnostic/wdbc.data')
X, y = wdbc.iloc[:, 2:].values, wdbc.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, shuffle=True, stratify=y)

print('0' * 100)
# print(len(X_train))
import numpy as np

param_range = [np.power(10, i * 1.0) for i in range(-3, 3)]
# K 近邻算法
param_grid = [
    {'svc__C': param_range, 'svc__kernel': ['linear']},
    {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf']},
]
gs_svc = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=2, refit=True)
score_svc=cross_val_score(gs_svc,X_train,y_train,scoring='accuracy',cv=5)
print(np.mean(score_svc),np.std(score_svc))

from sklearn.neighbors import KNeighborsClassifier

# K 近邻算法
pipe_knn = make_pipeline(StandardScaler(), PCA(n_components=6), KNeighborsClassifier())
param_grid = [
    {'kneighborsclassifier__n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10]},
    {'kneighborsclassifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
]
gs_knn = GridSearchCV(estimator=pipe_knn, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=10, refit=True)
score_knn=cross_val_score(gs_knn,X_train,y_train,scoring='accuracy',cv=5)
print(np.mean(score_knn),np.std(score_knn))

#决策树
from sklearn.tree import DecisionTreeClassifier
pipe_tree = make_pipeline(StandardScaler(), PCA(n_components=6), DecisionTreeClassifier())
param_grid = [
    {'decisiontreeclassifier__max_depth': [i for i in range(1, 8)] + [None]},
    {'decisiontreeclassifier__criterion': ["gini", "entropy"]}
]
gs_tree= GridSearchCV(estimator=pipe_tree, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=10, refit=True)
score_tree=cross_val_score(gs_tree,X_train,y_train,scoring='accuracy',cv=5)
print(np.mean(score_tree),np.std(score_tree))

# print('0' * 100)
# clf = grid_result.best_estimator_  # 最佳学习器
# print(clf)
# print('1' * 100)
# print(clf.score(X_test, y_test))
# print('2' * 100)
# means = grid_result.cv_results_['mean_test_score']
# params = grid_result.cv_results_['params']
# print(type(grid_result.cv_results_))
#
# print('3'*100)
# for mean, param in zip(means, params):
#     print('%f with : %r' % (mean, param))
