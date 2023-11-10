# -*- coding: utf-8 -*-
# @Time    : 2023/10/26 19:07
# @Author  : nanji
# @Site    : 
# @File    : test05.py
# @Software: PyCharm 
# @Comment :
import numpy as np

import matplotlib.pyplot as plt

l1_ratio, l2_ratio = 5, 5
max_epochs, eps = 200, 1e-15

obj_function = lambda x: (x - 1) ** 2  # 目标函数
obj_func_reg2 = lambda x: (x - 1) ** 2 + l2_ratio * x ** 2  # 目标函数 + L2 正则化
obj_func_reg1 = lambda x: (x - 1) ** 2 + l1_ratio * np.abs(x)  # 目标函数+L1正则化
grad_func = lambda x: 2 * (x - 1)  # 梯度


# grad_func_reg2 = lambda x: 2 * (x - 1) +2*l2_ratio*x  # 梯度
# grad_func_reg1 = lambda x: 2 * (x - 1)  +l1_ratio*np.sign(x)# 梯度


def gradient_descent_reg(x0, grad_fun, obj_fun_l2=None, obj_fun_l1=None):
    '''
    带有正则化的梯度下降法，求解单变量函数的极值问题
    :param x0: 初值
    :param grad_fun: 梯度函数
    :param obj_fun_l2: L2正则化目标函数
    :param obj_fun_l1: L1正则化目标函数
    :return:
    '''
    min_val = x0  # 初值，标记最小值点
    optional_values = [min_val]  # 记录最小值 更新
    alpha = 0.2  # 学习率
    if obj_fun_l2 is not None and obj_fun_l1 is None:  # L2 正则化
        for epoch_ in range(max_epochs):
            alpha *= 0.98
            min_val = min_val - alpha * (grad_fun(min_val) + 2 * l2_ratio * min_val)
            optional_values.append(min_val)
            if np.abs(optional_values[-1] - optional_values[-2]) <= eps:
                break
    elif obj_fun_l1 is not None and obj_fun_l2 is None:  # L1 正则化
        for epoch_ in range(max_epochs):
            alpha *= 0.98
            min_val = min_val - alpha * (grad_fun(min_val) + l1_ratio * np.sign(min_val))
            optional_values.append(min_val)
            if np.abs(optional_values[-1] - optional_values[-2]) <= eps:
                break
    else:  # 不采用正则化
        for epoch_ in range(max_epochs):
            alpha *= 0.98
            min_val = min_val - alpha * grad_fun(min_val)
            optional_values.append(min_val)
            if np.abs(optional_values[-1] - optional_values[-2]) <= eps:
                break
    return min_val, optional_values


def plt_extreme_point(x, y, x_min, y_min, style=None, lab=None):
    '''
    可视化目标函数和极值点
    :param x:
    :param y:
    :param x_min:
    :param y_min:
    :param style:
    :param labl:
    :return:
    '''
    if style is None:
        # plt.plot(x, y, 'k-', lw=1, label='Object Func +' + lab)  #
        # plt.plot(x_min, y_min, 'ko', label=lab.split(',')[0] + ", Minimum is $%.6f$" % x_min)
        plt.plot(x, y, lw=1, label='Object Func +' + lab)  #
        plt.plot(x_min, y_min, label=lab.split(',')[0] + ", Minimum is $%.6f$" % x_min)
    else:
        plt.plot(x, y, style[0], label='Object Func +' + lab)  #
        plt.plot(x_min, y_min, style[1], \
                 label=lab.split(',')[0] + ", Minimum is $%.6f$" % x_min)
    plt.legend(frameon=False)
    plt.grid(ls=":")
    plt.xlabel(r"$X$", fontdict={"fontsize": 12})
    plt.ylabel(r"$Y$", fontdict={"fontsize": 12})
    plt.title("Gradient Descent with Regularization ", fontdict={"fontsize": 14})


def plt_optimal_values(opt_sols, lab):
    '''
    可视化最优解搜索过程
    :param opt_sols:
    :param lab:
    :return:
    '''
    plt.figure(figsize=(7, 5))
    plt.plot(opt_sols, 'k-', lw=1)
    plt.grid(ls=":")
    plt.xlabel(r'Number of iterations', fontdict={'fontsize': 12})
    plt.ylabel(r'Optimal Solution', fontdict={'fontsize': 12})
    plt.title("The Optimal Solution Changes Curve by %s Regularization"  #
              % (lab), fontdict={"fontsize": 12})
    # plt.show()


if __name__ == '__main__':
    # X = np.linspace(-5, 5, 300)
    # y = obj_function(X)
    # X_l1 = obj_func_reg1(X)
    # X_l2 = obj_func_reg2(X)
    X0 = 3
    # min_val, optional_values = gradient_descent_reg(X0, grad_func, obj_func_reg1)
    # min_y = obj_function(min_val)
    # y0 = obj_function(X0)
    # print(min_val)
    # print(min_y)
    X = np.linspace(-2, 2, 100)
    # obj_func
    y = obj_function(X)
    min_val, optional_values = gradient_descent_reg(X0, grad_func)
    min_y = obj_function(min_val)
    # obj_func_reg2
    y_l2 = obj_func_reg2(X)
    min_val_l2, optional_values_l2 = gradient_descent_reg(X0, grad_func, obj_fun_l2=grad_func)
    min_y_l2 = obj_func_reg2(min_val_l2)
    # obj_func_reg1
    y_l1 = obj_func_reg1(X)
    min_val_l1, optional_values_l1 = gradient_descent_reg(X0, grad_func, obj_fun_l1=grad_func)
    min_y_l1 = obj_func_reg1(min_val_l1)
    plt.figure(figsize=(7, 5))
    plt_extreme_point(X, y, min_val, min_y, style=["g-","cs"],lab="obj_func")
    plt_extreme_point(X, y_l2, min_val_l2, min_y_l2,style=["k-","ro"], lab="obj_func_reg2")
    plt_extreme_point(X, y_l1, min_val_l1, min_y_l1,style=["r-","k*"],lab="obj_func_reg1")
    print('0' * 100)
    plt.figure(figsize=(7, 5))
    plt_optimal_values(optional_values, "obj_func ")
    plt.figure(figsize=(7, 5))
    plt_optimal_values(optional_values_l2, "grad_func_reg2 ")
    plt.figure(figsize=(7, 5))
    plt_optimal_values(optional_values_l1, "grad_func_reg1 ")
    plt.show()
