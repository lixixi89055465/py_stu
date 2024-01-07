# -*- coding: utf-8 -*-
# @Time : 2024/1/6 15:13
# @Author : nanji
# @Site : https://blog.csdn.net/wyssailing/article/details/100626703
# @File : testMinMaxScaler.py
# @Software: PyCharm 
# @Comment :

from sklearn import preprocessing
import numpy as np

X = np.array([
	[1., -1, 2],
	[2., 0, 0],
	[0., 1, -1],
])
min_max_scaler = preprocessing.MinMaxScaler()
x_minmax = min_max_scaler.fit_transform(X)
print(x_minmax)
