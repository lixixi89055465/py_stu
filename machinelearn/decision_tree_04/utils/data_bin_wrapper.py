# -*- coding: utf-8 -*-
# @Time    : 2023/9/24 下午11:29
# @Author  : nanji
# @Site    : 
# @File    : data_bin_wrapper.py
# @Software: PyCharm 
# @Comment :

import numpy as np


class DataBinWrapper:
    '''
    连续特征数据的离散化，分箱(分段)操作,根据用户传参_bins,计算分位数，以分位数（分箱）分段
    然后根据样本特征取值所在的区间段（哪个箱）位置索引标记当前值
    1.fit(x)根据样本进行分箱
    2.transform(x)根据已存在的箱，把数据分成max_bins类
    '''

    def __init__(self, max_bins=10):
        self.max_bins = max_bins  # 分箱数
        self.XrangeMap = None  # 箱（区间数）

    def fit(self, x_samples):
        '''

        :param x_samples: 样本(二维数组 n*k )，或是一个特征属性的数据(二维数组n*1)
        :return:
        '''
        if x_samples.ndim == 1:  # 一个特征属性，转换为二维数组
            n_features = 1
            x_samples = x_samples[:, np.newaxis]  # 添加一个轴，转换为二维数组
        else:
            n_features = x_samples.shape[1]

        # 构建分箱，区间段
        self.XrangeMap = [[] for _ in range(n_features)]
        for idx in range(n_features):
            x_sorted = sorted(x_samples[:, idx])  # 按特征索引取值，并从小到大排序
            for bin in range(1, self.max_bins):
                p = (bin / self.max_bins) * 100
                p_val = np.percentile(x_sorted, p)
                self.XrangeMap[idx].append(p_val)
            self.XrangeMap[idx]=sorted(list(set(self.XrangeMap[idx])))
    def transform(self,x_samples,XrangeMap=None):
        '''
        根据已存在的箱，把数据分成max_bins类
        :param x_samples: 样本(二维数组n*k),或一个特征属性的数据（二维数组n*1)
        :return:
        '''
        if x_samples.ndim==1:
            if XrangeMap is not None:
                return np.asarray(np.digitize(x_samples,XrangeMap)).reshape(-1)
            else:
                return np.asarray(np.digitize(x_samples,self.XrangeMap[0])).reshape(-1)
        else:
            return np.asarray([np.digitize(x_samples[:,i],self.XrangeMap[i]) for i in range(x_samples.shape[1])]).T





if __name__ == '__main__':
    # x = np.random.randint(10, 80, 20)
    x = np.random.randn(10,5)
    dbw = DataBinWrapper(max_bins=5)
    print(x)
    dbw.fit(x)
    print(dbw.XrangeMap)
    print('1'*100)
    print(dbw.transform(x))
