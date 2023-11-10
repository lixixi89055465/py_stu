# -*- coding: utf-8 -*-
# @Time    : 2023/11/6 21:15
# @Author  : nanji
# @Site    : 
# @File    : sklearn_agens.py
# @Software: PyCharm 
# @Comment :
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot  as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import seaborn as sns

if __name__ == '__main__':
    X = pd.read_csv('../data/consumption_data.csv')
    X = StandardScaler().fit_transform(X)
    ac = AgglomerativeClustering(n_clusters=3, linkage='complete')
    ac.fit(X)
    labels = ac.labels_
    # 可视化和密度估计
    title = ['R index', 'F index', 'M index']
    plt.figure(figsize=(7, 10))
    cluster_k = np.unique(labels)
    for f in range(X.shape[1]):
        plt.subplot(311 + f)
        for c in cluster_k:
            sns.kdeplot(X[labels == c][:, f])
        plt.grid()
        plt.title(title[f])
    plt.savefig('sklearn_agens.py.png')
    plt.show()
