# -*- coding: utf-8 -*-
# @Time    : 2023/10/1 17:01
# @Author  : nanji
# @Site    : 
# @File    : square_error_utils.py
# @Software: PyCharm 
# @Comment : 回归决策树——回归树建立
import numpy as np
class SquareErrorUtils:
    '''
    平方误差最小化尊则选择其中最优的一个作为切分点
    '''

    @staticmethod
    def _set_sample_weight(sample_weight, n_sample):
        '''
        扩展到集成学习，此处为样本权重的设置

        :param sample_weight:各样本的权重
        :param n_sample: 样本量
        :return:
        '''
        if sample_weight is None:
            sample_weight = np.asarray([1.0] * n_sample)
        return sample_weight

    @staticmethod
    def square_error(y, sample_weight):
        '''
        平方误差
        :param y:  当前划分趋于的目标值集合
        :param sample_weight:  当前样本的权重
        :return:
        '''
        y = np.asarray(y)
        return np.sum((y - np.mean(y)) ** 2 * sample_weight)

    def cond_square_error(self, x, y, sample_weight):
        '''
        计算根据特征x划分的趋于中y的误差值
        :param x:  某个特征划分区域所包含的样本
        :param y: x对应的目标值
        :param sample_weight: 当前x的权重
        :return:
        '''
        x, y = np.asarray(x), np.asarray(y)
        error = 0.0
        for x_val in set(x):
            x_idx = np.where(x == x_val)  # 按区域计算 误差
            new_y = y[x_idx]  # 对应区域的目标值
            new_sample_weight = sample_weight[x_idx]
            error += SquareErrorUtils.square_error(new_y, new_sample_weight)
        return error

    def square_error_gain(self, x, y, sample_weight=None):
        '''
        平方误差带来的增益值
        :param x:  某特征划分区域后所对应的样本标记
        :param y: x对应的目标值
        :param sample_weight: 各样本权重，方便后续集成学习
        :return:
        '''
        sample_weight = self._set_sample_weight(sample_weight, len(x))
        return SquareErrorUtils.square_error(y, sample_weight) - self.cond_square_error(x, y, sample_weight)
