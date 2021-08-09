# coding: utf-8
# @Time    : 2020-07-22 22:35
# @Author  : lixiang
# @File    : K-临近-01.py


#导包
import numpy as np
import pandas  as pd
from pandas import DataFrame,Series
from sklearn.neighbors import KNeighborsClassifier #k邻近算法模型

#使用datasets创建数据
import sklearn.datasets as datasets
iris = datasets.load_iris()

feature = iris['images']
target = iris['target']

#将样本打乱，符合真实情况

np.random.seed(1)
np.random.shuffle(feature)
np.random.seed(1)
np.random.shuffle(target)

#训练数据
x_train = feature[:140]
y_train = target[:140]
#测试数据
x_test = feature[-10:]
y_test =target[-10:]

#实例化模型对象&训练模型
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)
knn.score(x_train,y_train)

print('预测分类：',knn.predict(x_test))
print('真实分类：',y_test)