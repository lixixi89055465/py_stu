# -*- coding: utf-8 -*-
# @Time    : 2023/9/23 上午10:29
# @Author  : nanji
# @Site    : 
# @File    : lda_multi_dim_reduction.py
# @Software: PyCharm 
# @Comment :
import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer,load_wine,make_classification
from machinelearn.linear_model_03.lda_lecture.lda_multi_dim_reduction import LDAMulti_DimReduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
# iris = load_iris()
# iris = load_breast_cancer()
# iris = load_wine()
# X, y = iris.data, iris.target
X,y=make_classification(n_samples=2000,n_features=20,n_informative=3,n_classes=5,
                        n_redundant=0,n_clusters_per_class=1,class_sep=2,random_state=42)

X=StandardScaler().fit_transform(X)

lda = LDAMulti_DimReduction(n_components=3)
lda.fit(X, y)

print(X.shape)
x_skl = lda.transform(X)

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 10))
plt.subplot(221)
plt.scatter(x_skl[:, 0], x_skl[:, 1], marker='o', c=y)
plt.xlabel('PC1', fontdict={'fontsize': 12})
plt.ylabel('PC2', fontdict={'fontsize': 12})
plt.title('LDA Dimension Reduction (Myself)', fontdict={'fontsize': 14})
plt.grid(ls=':')


plt.subplot(222)
plt.scatter(x_skl[:, 1], x_skl[:, 2], marker='o', c=y)
plt.xlabel('PC2', fontdict={'fontsize': 12})
plt.ylabel('PC3', fontdict={'fontsize': 12})
plt.title('LDA Dimension Reduction (Myself)', fontdict={'fontsize': 14})
plt.grid(ls=':')


lda = LinearDiscriminantAnalysis(n_components=3)
lda.fit(X, y)
print(X.shape)
x_skl = lda.transform(X)

plt.subplot(223)
plt.scatter(x_skl[:, 0], x_skl[:, 1], marker='o', c=y)
plt.xlabel('PC1', fontdict={'fontsize': 12})
plt.ylabel('PC2', fontdict={'fontsize': 12})
plt.title('LDA Dimension Reduction (Myself)', fontdict={'fontsize': 14})
plt.grid(ls=':')

plt.subplot(224)
plt.scatter(x_skl[:, 1], x_skl[:, 2], marker='o', c=y)
plt.xlabel('PC2', fontdict={'fontsize': 12})
plt.ylabel('PC3', fontdict={'fontsize': 12})
plt.title('LDA Dimension Reduction (Myself)', fontdict={'fontsize': 14})
plt.grid(ls=':')


plt.show()
print('3'*100)
print(lda.explained_variance_ratio_)
