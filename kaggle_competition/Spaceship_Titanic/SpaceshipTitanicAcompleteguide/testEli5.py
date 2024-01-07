# -*- coding: utf-8 -*-
# @Time : 2024/1/6 16:15
# @Author : nanji
# @Site :  https://zhuanlan.zhihu.com/p/385424638
# @File : testEli5.py
# @Software: PyCharm 
# @Comment : 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd

mpg = pd.read_csv('../data/mpg.csv').dropna()
mpg.drop('name', axis=1, inplace=True)
print(mpg.columns)

X_train, X_test, y_train, y_test = train_test_split(
	mpg.drop('origin', axis=1), \
	mpg['origin'], \
	test_size=0.2, \
	random_state=121)

# model training
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
import eli5

eli5.show_weights(clf, feature_names=list(X_test.columns))
