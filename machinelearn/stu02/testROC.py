# -*- coding: utf-8 -*-
# @Time    : 2023/10/7 17:23
# @Author  : nanji
# @Site    : 
# @File    : testROC.py
# @Software: PyCharm 
# @Comment :  3. 性能度量——ROC与AUC+二分类
import pandas as pd

breatcaner = pd.read_csv('../data/breast-cancer.csv', header=None).iloc[:, 1:]
print('0' * 100)
print(breatcaner.head())
