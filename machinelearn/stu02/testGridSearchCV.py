# -*- coding: utf-8 -*-
# @Time    : 2023/9/1 15:35
# @Author  : nanji
# @Site    : 
# @File    : testGridSearchCV.py
# @Software: PyCharm 
# @Comment :
from sklearn.model_selection import GridSearchCV, train_test_split
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
print(len(X_train))
import numpy as np

param_range = [np.power(10, i * 1.0) for i in range(-3, 3)]
param_grid = [
    {'svc__C': param_range, 'svc__kernel': ['linear']},
    {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf']},
]
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=10, refit=True)
grid_result = gs.fit(X_train, y_train) # type:GridSearchCV
grid_result.best_params_

print(type(grid_result))
print('Best: %f using %s ' % (grid_result.best_score_, grid_result.best_params_))

print('0' * 100)
clf = grid_result.best_estimator_  # 最佳学习器
print(clf)
print('1' * 100)
print(clf.score(X_test, y_test))
print('2' * 100)
means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']
print(type(grid_result.cv_results_))

print('3'*100)
for mean, param in zip(means, params):
    print('%f with : %r' % (mean, param))


