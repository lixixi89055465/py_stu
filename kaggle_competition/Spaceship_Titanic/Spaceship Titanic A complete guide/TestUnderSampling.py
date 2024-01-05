# -*- coding: utf-8 -*-
# @Time : 2024/1/5 23:33
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/613087774
# @File : TestUnderSampling.py
# @Software: PyCharm 
# @Comment :

from sklearn.datasets import make_classification
import collections
from imblearn.under_sampling import RandomUnderSampler

# 生成样本数据
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], \
						   random_state=42)
print(list(collections.Counter(y).items()))
# 实例化 RandomUnderSampler类
rus = RandomUnderSampler(random_state=42)
# 对样本进行欠采样
X_resampled, y_resampled = rus.fit_resample(X,y)
print(list(collections.Counter(y_resampled).items()))

