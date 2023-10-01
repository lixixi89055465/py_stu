# -*- coding: utf-8 -*-
# @Time    : 2023/10/1 21:26
# @Author  : nanji
# @Site    : 
# @File    : test_decision_treee.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import matplotlib.pyplot as plt
from machinelearn.decision_tree_04.decision_tree_R import DecisionTreeRegression

obj_fun = lambda x: np.sin(x)
np.random.seed(1)
n = 100
x = np.linspace(0, 10, n)
target = obj_fun(x) + 0.3 * np.random.rand(n)
data = x[:, np.newaxis]  # 二维数组
# print('0' * 100)
# print(target)

# tree = DecisionTreeRegression(max_bins=50, max_depth=20)
tree = DecisionTreeRegression(max_bins=50)
tree.fit(data, target)
x_test = np.linspace(0, 10, 200)
y_test = tree.predict(x_test[:, np.newaxis])

plt.figure(figsize=(7, 5))
plt.scatter(data, target, s=15, c='k', label='Raw data')
plt.plot(x_test, y_test, 'r--', lw=1.5, label='Fit Model')
plt.legend(frameon=False)
plt.show()
