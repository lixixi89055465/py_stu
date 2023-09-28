# -*- coding: utf-8 -*-
# @Time    : 2023/9/24 下午7:20
# @Author  : nanji
# @Site    : 
# @File    : test_entropy.py
# @Software: PyCharm 
# @Comment :
import pandas as pd
from machinelearn.decision_tree_04.utils.entropy_utils import EntropyUtils
from machinelearn.decision_tree_04.utils.data_bin_wrapper import DataBinWrapper
data = pd.read_csv('../../data/watermelon.csv').iloc[:, 1:]
y = data.iloc[:, -1]
feature_names = data.columns[:-1]
print(feature_names)
print('0' * 100)
ent_obj = EntropyUtils()
for feat in feature_names:
    print(feat, ":", ent_obj.info_gain(data.loc[:, feat], y))

print('=' * 100)
for feat in feature_names:
    print(feat, ":", ent_obj.info_gain_rate(data.loc[:, feat], y))

print('=' * 100)
for feat in feature_names:
    print(feat, ":", ent_obj.gini_gain(data.loc[:, feat], y))

print('1'*100)
import numpy as np
x1=np.asarray(data.loc[:,['密度','含糖率']])
print(x1)

print('2'*100)
dbw=DataBinWrapper(max_bins=8)
dbw.fit(x1)
print('3'*100)
print(dbw.transform(x1))
