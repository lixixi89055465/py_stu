# -*- coding: utf-8 -*-
# @Time    : 2023/10/2 15:28
# @Author  : nanji
# @Site    : 
# @File    : distUtils.py
# @Software: PyCharm 
# @Comment :
import numpy as np


class DistanceUtils:
    '''
    距离度量的工具类，此处仅实现闵可夫斯基距离
    '''

    def __init__(self, p=2):
        self.p = p  # 模式欧式距离，p=np.inf 是切比雪夫距离

    def distance_func(self, xi, xj):
        '''
        特征空间中两个样本实例的距离计算
        :param xi: k维空间某个样本实例
        :param xj: k维空间某个样本实例
        :return:
        '''
        if self.p == 1 or self.p == 2:
            return (((np.abs(xi - xj)) ** self.p).sum() ** (1.0 / self.p))
        elif self.p == np.inf:
            return np.max(np.abs(xi - xj))
        else:
            raise ValueError('目前仅仅支持p=1,p=2或p=np.inf三种距离 ')
