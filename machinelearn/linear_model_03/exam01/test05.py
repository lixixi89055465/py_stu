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

obj_function = lambda x: (x - 1) * 2  # 目标函数
obj_func_reg2 = lambda x: (x - 1) ** 2 + l2_ratio * x ** 2  # 目标函数 + L2 正则化
obj_func_reg1 = lambda x: (x - 1) ** 2 + l1_ratio * np.abs(x)  # 目标函数+L1正则化
grad_func = lambda x: 2 * (x - 1)  # 梯度


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
            min_val = min_val - alpha * (grad_func(min_val) + 2 * l2_ratio * min_val)
            optional_values.append(min_val)
            if np.abs(optional_values[-1] - optional_values[-2]) <= eps:
                break
    elif obj_fun_l1 is not None and obj_fun_l2 is None:  # L1 正则化
        for epoch_ in range(max_epochs):
            alpha *= 0.98
            min_val = min_val - alpha * (grad_func(min_val) + 2 * l1_ratio * np.sign(min_val))
            optional_values.append(min_val)
            if np.abs(optional_values[-1] - optional_values[-2]) <= eps:
                break
    else:  # 不采用正则化
        for epoch_ in range(max_epochs):
            alpha *= 0.98
            min_val = min_val - alpha * grad_func(min_val);
            optional_values.append(min_val)
            if np.abs(optional_values[-1] - optional_values[-2] <= eps):
                break
    return min_val, optional_values


