# -*- coding: utf-8 -*-
# @Time    : 2023/9/1 10:14
# @Author  : nanji
# @Site    : 
# @File    : testStratifiedKFold.py
# @Software: PyCharm 
# @Comment :
from sklearn.model_selection import StratifiedKFold

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

print('0' * 100)
wdbc = pd.read_csv('../breast+cancer+wisconsin+diagnostic/wdbc.data')
X, y = wdbc.iloc[:, 2:].values, wdbc.iloc[:, 1].values
print(wdbc.info())
y = LabelEncoder().fit_transform(y)
pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=6),
                        LogisticRegression())
# 划分数据集为训练集和测试集，比例8:2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1, shuffle=True)
# K_fold = StratifiedKFold(n_splits=10).split(X_train, y_train)
# scores = []
# import numpy as np
#
# for i, (train, test) in enumerate(K_fold):
#     pipe_lr.fit(X_train[train], y_train[train])
#     score = pipe_lr.score(X_train[test], y_train[test])
#     scores.append(score)
#     print('Fold: %2d,class dist :%s, Acc:%.3f' % (i + 1, np.bincount(y_train[train]), score))
K_fold = StratifiedKFold(n_splits=10).split(X_train, y_train)
scores = []
import numpy as np

for i, (train, test) in enumerate(K_fold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold : %d\t%s\t%.3f' % (i + 1, np.bincount(y_train[train]), score))
print('\n CV accuracy : %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
