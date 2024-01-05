# -*- coding: utf-8 -*-
# @Time : 2024/1/5 22:00
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/613087774
# @File : testRandomSampling.py
# @Software: PyCharm 
# @Comment : 

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
from sklearn.model_selection import train_test_split

# 随机采样  Random sampling
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# print(X_train[:10])

# 分层采样 Stratified Sampling
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
from sklearn.model_selection import StratifiedShuffleSplit

for train_index, test_index in sss.split(X, y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

# 过采样 oversampling
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 生成样本数据
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=2, \
						   n_classes=2, n_repeated=2, n_clusters_per_class=2, \
						   weights=[0.1, 0.9], flip_y=0, class_sep=2, random_state=10
						   )
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, \
													random_state=42)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(x_train, y_train)
print('0' * 100)
print('采样前训练集中类别0的数目：', sum(y_train == 0))
print('采样前训练集中类别1的数目：', sum(y_train == 0))
print('采样后训练集中类别0的数目：', sum(y_train_res == 0))
print('采样后训练集中类别1的数目：', sum(y_train_res == 1))

# 欠采样 UnderSampling
