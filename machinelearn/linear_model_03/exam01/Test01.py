# -*- coding: utf-8 -*-
# @Time    : 2023/10/24 13:13
# @Author  : nanji
# @Site    : 
# @File    : Test01.py
# @Software: PyCharm 
# @Comment :

import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
# from machinelearn.decision_tree_04.decision_tree_R import DecisionTreeRegression
from sklearn.tree import DecisionTreeRegressor
from machinelearn.ensemble_learning_08.gradient.gradientboosting_r import GradientBoostRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def gradient_descent_train(x0, alpha, iter_max, eta, obj_fun, grad_fun):
    '''
   梯度下降法求解函数，以及多个极小值函数情况。
   alpha过小迭代步数过多，容易局部收敛
    :param x0:
    :param alpha:
    :param iter_max:
    :param eta:
    :param obj_fun:
    :param grad_fun:
    :return:
    '''
    xi = np.linspace(-3.6, 3.6, 100)  # 模拟值
    yi = obj_fun(xi)  # 模拟函数
    min_x, min_y = x0, obj_fun(x0)  # 初始值x0
    x_ = [min_x]
    y_ = [min_y]
    iter_ = 0  # 魂环次数
    for iter_ in range(iter_max):
        min_x = min_x - alpha * grad_fun(min_x)
        x_.append(min_x)
        y_.append(obj_fun(min_x))
        if np.abs(grad_fun(min_x)) <= eta:  # 梯度精度控制
            break
    plt_loss(xi, yi, x_, y_, iter_ + 1, alpha, x0)  # 绘制图像


def plt_loss(xi, yi, x_, y_, iter, alpha, x0):
    '''
    绘制函数图像和迭代下降值的变化过程
    :param xi:
    :param yi:
    :param x_:
    :param y_:
    :param iter:
    :param alpha:
    :param x0:
    :return:
    '''
    plt.figure(figsize=(7,5))
    plt.plot(xi, yi, 'r--', lw=1.5, label='Object Function')
    plt.plot(x_[:50], y_[:50], 'k.-', label='Minimum Seeking')
    plt.plot(x_[0], y_[0], 'cs', label='Initial Value')
    plt.plot(x_[-1], y_[-1], 'bo', label='Local or Global Minimum')
    plt.legend(fontsize=12)
    plt.xlabel(r"$X$", fontdict={'fontsize': 12})
    plt.ylabel(r"$Y$", fontdict={'fontsize': 12})
    print('迭代 %d次后，最小值点%.15f,极小值%.15f.' % (iter, x_[-1], y_[-1]))


obj_function = lambda x: x ** 2  # 目标函数
grad_function = lambda x: 2 * x  # 梯度
iter_max = 10000
# x0, alpha = 3.5, 0.2
# gradient_descent_train(x0=x0, alpha=alpha, iter_max=iter_max, eta=1e-8, obj_fun=obj_function, grad_fun=grad_function)
# x0, alpha = 3.5, 0.01
# gradient_descent_train(x0=x0, alpha=alpha, iter_max=iter_max, eta=1e-8, obj_fun=obj_function, grad_fun=grad_function)
# x0, alpha = 3.5, 0.9
# gradient_descent_train(x0=x0, alpha=alpha, iter_max=iter_max, eta=1e-8, obj_fun=obj_function, grad_fun=grad_function)


obj_function = lambda x: 3 * np.exp(-2 * x / 3) * np.sin(1 + 2 * x) / 8  # 目标函数
grad_function = lambda x: (np.exp(-2 * x / 3) * (3 * np.cos(1 + 2 * x) - np.sin(1 + 2 * x))) / 4  # 梯度
x0, alpha = 3, 0.1
gradient_descent_train(x0=x0, alpha=alpha, iter_max=iter_max, eta=1e-8, obj_fun=obj_function, grad_fun=grad_function)
x0, alpha = 0.2, 0.1
gradient_descent_train(x0=x0, alpha=alpha, iter_max=iter_max, eta=1e-8, obj_fun=obj_function, grad_fun=grad_function)
x0, alpha = -3, 0.1
gradient_descent_train(x0=x0, alpha=alpha, iter_max=iter_max, eta=1e-8, obj_fun=obj_function, grad_fun=grad_function)
x0, alpha = -3.5, 0.1
gradient_descent_train(x0=x0, alpha=alpha, iter_max=iter_max, eta=1e-8, obj_fun=obj_function, grad_fun=grad_function)

plt.show()
