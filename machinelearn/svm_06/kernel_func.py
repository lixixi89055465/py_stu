# -*- coding: utf-8 -*-
# @Time    : 2023/10/4 13:51
# @Author  : nanji
# @Site    : 
# @File    : kernel_func.py
# @Software: PyCharm 
# @Comment :
import numpy as np


def linear():
    '''
    线性和函数
    :return:
    '''

    def _linear(x_i, x_j):
        return np.dot(x_i, x_j)

    return _linear


def poly(degree=3, coef=1.0):
    '''
    多项式和函数
    :param degree: 阶次
    :param coef: 常数项
    :return:
    '''

    def _poly(x_i, x_j):
        return np.power(np.dot(x_i, x_j) + coef,degree)

    return _poly

def rbf(gamma=1.0):
    '''
    高斯核函数
    :param gamma: 超参数
    :return:
    '''
    def _rbf(x_i,x_j):
        x_i,x_j=np.asarray(x_i),np.asarray(x_j)
        if x_i.ndim<=1:
            return np.exp(-np.dot(x_i-x_j,x_i-x_j))/(2*gamma**2)
        else:
            return np.exp(-np.multiply(x_i-x_j,x_i-x_j).sum(axis=1)/(2*gamma**2))

    return _rbf
