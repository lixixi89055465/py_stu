# -*- coding: utf-8 -*-
# @Time    : 2023/8/31 14:17
# @Author  : nanji
# @Site    : 
# @File    : testWSKXRXA.py
# @Software: PyCharm 
# @Comment :
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler  # 标签编码和标准化

wdbc = pd.read_csv('../breast+cancer+wisconsin+diagnostic/wdbc.data')
# 获取样本数据集和目标标签集
# print(wdbc.info())

X, y = wdbc.iloc[:, 2:], wdbc.iloc[:, 1]
print('0' * 100)
X = StandardScaler().fit_transform(X)
label_en = LabelEncoder()
y = label_en.fit_transform(y)

# 查看类别标签，以及转换后的对应值
print(label_en.classes_)
print(label_en.transform(['B', 'M']))
from sklearn.decomposition import PCA

pca = PCA(n_components=6).fit(X)
evr = pca.explained_variance_ratio_  # 解释方差比，即各主成分贡献率
print('各主成分贡献率 ', evr, '\n累计贡献率 ', np.cumsum(evr))

X_pca = pca.transform(X)
print('1' * 100)
print(X_pca[:5, :])
import matplotlib.pyplot as plt

plt.figure(figsize=(21, 5))
X_b, X_m = X_pca[y == 0], X_pca[y == 1]  # 分别获取良性和恶行肿瘤
for i in range(3):
    plt.subplot(131 + i)
    plt.plot(X_b[:, i * 2], X_b[:, 2 * i + 1], 'ro', markersize=3, label='benign')
    plt.plot(X_m[:, i * 2], X_m[:, 2 * i + 1], 'bx', markersize=5, label='malignant')
    plt.legend(frameon=False)
    plt.grid(ls=':')
    plt.xlabel(str(i * 2 + 1) + "th principal component", fontdict={'fontsize': 12})
    plt.ylabel(str(i * 2 + 2) + "th principal component", fontdict={'fontsize': 12})
    plt.title('Each category of data after dim reductino by PCA')
plt.show()

print('2' * 100)
