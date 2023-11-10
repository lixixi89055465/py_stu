# -*- coding: utf-8 -*-
# @Time    : 2023/9/1 9:52
# @Author  : nanji
# @Site    : 
# @File    : testPipline.py
# @Software: PyCharm 
# @Comment :

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

print('0' * 100)
wdbc = pd.read_csv('../breast+cancer+wisconsin+diagnostic/wdbc.data')
X, y = wdbc.iloc[:, 2:], wdbc.iloc[:, 1]
print(wdbc.info())
pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=6),
                        LogisticRegression())
# 划分数据集为训练集和测试集，比例8:2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1, shuffle=True)

pipe_lr.fit(X_train, y_train)
y_pred=pipe_lr.predict(X_test)
print('1'*100)
print('Test accuracy :%.3f'%pipe_lr.score(X_test,y_test))
