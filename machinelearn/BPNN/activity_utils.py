# -*- coding: utf-8 -*-
# @Time    : 2023/11/11 10:22
# @Author  : nanji
# @Site    : 
# @File    : activity_utils.py
# @Software: PyCharm 
# @Comment :
import numpy as np
from scipy.special import expit


def activity_functions(type):
    '''
    激活函数工具类
    :param type:
    :return:
    '''

    def sigmoid(x):
        x = np.array(x, dtype=np.float)
        return expit(x)

    def diff_sigmoid(x):
        return x * (1 - x)

    def tanh(x):
        return np.tanh(x)

    def diff_tanh(x):
        return 1 - np.tanh(x) ** 2

    if type == 'sigmoid':
        return sigmoid, diff_sigmoid
    elif type == 'tanh':
        return tanh, diff_tanh
    else:
        raise AttributeError("激活函数选择有误...")
