# -*- coding: utf-8 -*-
# @Time    : 2023/8/31 15:08
# @Author  : nanji
# @Site    : 
# @File    : testwdbc.py
# @Software: PyCharm 
# @Comment : 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler  # 标签编码和标准化

wdbc = pd.read_csv('../breast+cancer+wisconsin+diagnostic/wdbc.data')
X, y = wdbc.iloc[:, 2:], wdbc.iloc[:, 1]

# 获取样本数据集和目标标签集
X = StandardScaler().fit_transform(X)
label_en = LabelEncoder()
y = label_en.fit_transform(y)
# 查看类别标签，以及转换后的对应值
from sklearn.decomposition import PCA

pca = PCA(n_components=6).fit(X)
X_pca = pca.transform(X)
acc_test_score, acc_train_score = [], []  #
for i in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, stratify=y, random_state=i,
                                                        shuffle=True)
    lg = LogisticRegression()
    lg.fit(X_train, y_train)
    y_pred = lg.predict(X_test)
    acc_test_score.append(accuracy_score(y_test, y_pred))
    acc_train_score.append(accuracy_score(y_train, lg.predict(X_train)))
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
plt.plot(acc_test_score, 'ro:', label='Test')
plt.plot(acc_train_score, 'ks--', label='train')
plt.grid(ls=':')
plt.legend(frameon=False)

plt.show()
