# -*- coding: utf-8 -*-
# @Time    : 2023/11/10 22:38
# @Author  : nanji
# @Site    : 
# @File    : Simple_Bpnn.py
# @Software: PyCharm 
# @Comment :
class SimpleNeuralNetwork:
    '''
    单层神经网络：无隐层
    '''

    def __init__(self, eta=1e-2, precision=None, gradient_method='SGD', \
                 optimizer_method=None, activity_fun='sigmoid', epochs=1000):
        '''
        :param eta: 学习率
        :param precision: 停机精度
        :param gradient_method: 梯度方法：SGD、BGD、MBGD　...
        :param optimizer_method: 优化函数：动量法、adagrad、adam...
        :param activity_fun: 激活函数，默认sigmoid
        :param epochs: 最大训练次数
        '''
        pass
