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
target = obj_fun(x) + 0.2 * np.random.rand(n)
data = x[:, np.newaxis]  # 二维数组
# print('0' * 100)
# print(target)

# tree = DecisionTreeRegression(max_bins=50, max_depth=10)
tree = DecisionTreeRegression(max_bins=50,max_depth=10)
tree.fit(data, target)
x_test = np.linspace(0, 10, 200)
y_test_pred = tree.predict(x_test[:, np.newaxis])
mse, r2 = tree.cal_mse_r2(obj_fun(x_test), y_test_pred)

plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.scatter(data, target, s=15, c='k', label='Raw data')
plt.plot(x_test, y_test_pred, 'r-', lw=1.5, label='Fit Model')
plt.xlabel('x',fontdict={'fontsize':12,'color':'b'})
plt.ylabel('y',fontdict={'fontsize':12,'color':'r'})
plt.grid(ls=':')
plt.legend(frameon=False)
plt.title('Regression Decision Tree(UnPrune) and MSE = %.5f R2= %.5f' % (mse, r2))

plt.subplot(122)
tree.prune(0.2)
y_test_pred=tree.predict(x_test[:,np.newaxis])
mse,r2=tree.cal_mse_r2(obj_fun(x_test),y_test_pred)
plt.scatter(data, target, s=15, c='k', label='Raw data')
plt.plot(x_test, y_test_pred, 'r-', lw=1.5, label='Fit Model')
plt.xlabel('x',fontdict={'fontsize':12,'color':'b'})
plt.ylabel('y',fontdict={'fontsize':12,'color':'r'})
plt.grid(ls=':')
plt.legend(frameon=False)
plt.title('Regression Decision Tree(UnPrune) and MSE = %.5f R2= %.5f' % (mse, r2))
plt.show()
