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
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston = load_boston()
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
tree = DecisionTreeRegression(max_bins=100)
tree.fit(X_train, y_train)
y_test_pred = tree.predict(X_test)
mse, r2 = tree.cal_mse_r2(y_test, y_test_pred)
idx=np.argsort(y_test)
plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.plot(y_test[idx], 'k-', lw=2, label='Test True values')
plt.plot(y_test_pred[idx], 'r-', lw=1.5, label='Test Predictive Values')
plt.xlabel('x', fontdict={'fontsize': 12, 'color': 'b'})
plt.ylabel('y', fontdict={'fontsize': 12, 'color': 'r'})
plt.grid(ls=':')
plt.legend(frameon=False)
plt.title('Regression Decision Tree(UnPrune) and MSE = %.5f R2= %.5f' % (mse, r2))

plt.subplot(122)
tree.prune(100)
y_test_pred = tree.predict(X_test)
mse, r2 = tree.cal_mse_r2(y_test, y_test_pred)

plt.plot(y_test[idx], 'k-', lw=2, label='Test True values')
plt.plot(y_test_pred[idx], 'r-', lw=1.5, label='Test Predictive Values')
plt.xlabel('x', fontdict={'fontsize': 12, 'color': 'b'})
plt.ylabel('y', fontdict={'fontsize': 12, 'color': 'r'})
plt.grid(ls=':')
plt.legend(frameon=False)
plt.title('Regression Decision Tree(Prune) and MSE = %.5f R2= %.5f' % (mse, r2))

plt.show()
