# -*- coding: utf-8 -*-
# @Time    : 2023/10/14 下午12:44
# @Author  : nanji
# @Site    : 
# @File    : EM01.py
# @Software: PyCharm 
# @Comment :  https://www.bilibili.com/video/BV1ku411s7Nk/?spm_id_from=333.788&vd_source=50305204d8a1be81f31d861b12d4d5cf

import numpy as np
import random, math
from scipy.stats import multinomial


# 定义高斯分布概率密度函数
def gaussian(x, mean, variance):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * variance)))
    return exponent / (math.sqrt(2 * math.pi * variance))


# EM 算法
def em_algorithm(data, num_clusters, num_iterations):
    # 随机初始化均值和方差
    means = [random.uniform(min(data), max(data)) for _ in range(num_clusters)]
    variances = [1.0] * num_clusters
    for _ in range(num_iterations):
        #  E步骤：计算每个样本属于每个簇的概率
        responsibilities = []
        for x in data:
            probabilities = [gaussian(x, mean, variance) for mean, variance in zip(means, variances)]
            total_probability = sum(probabilities)
            responsibilities.append([p / total_probability for p in probabilities])
        # M步骤：更新均值和方差
        for i in range(num_clusters):
            total_responsibility = sum(r[i] for r in responsibilities)
            means[i] = sum(responsibilities[j][i] * data[j] \
                           for j in range(len(data))) / total_responsibility
            variances[i] = sum(responsibilities[j][i] * (data[j] - means[i]) ** 2 \
                               for j in range(len(data))) / total_responsibility
    return means, variances


# 真实的高斯分布参数
true_means = [1.5, 2.5]
true_variances = [0.5, 0.3]
# 生成 符合真实高斯分布的样本数据
np.random.seed(1)
d = np.concatenate([np.random.normal(mean, math.sqrt(variance), size=100) \
                    for mean, variance in zip(true_means, true_variances)])
# 调用EM算法估计高斯分布参数
estimated_means, estimcated_variances = em_algorithm(d, num_clusters=2, num_iterations=700)
# 计算估计结果与真实结果之间的差异
mean_errors = np.abs(np.array(estimated_means) - np.array(true_means))
variance_errors = np.abs(np.array(estimcated_variances) - np.array(true_variances))

# 打印估计结果与真实结果之间的差异
print("均值误差", mean_errors)
print("方差误差 ", variance_errors)
