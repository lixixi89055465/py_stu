# -*- coding: utf-8 -*-
# @Time    : 2023/9/24 下午7:20
# @Author  : nanji
# @Site    : 
# @File    : test_entropy.py
# @Software: PyCharm 
# @Comment :
import pandas as pd
from machinelearn.decision_tree_04.entropy_utils import EntropyUtils
data=pd.read_csv('../../data/watermelon.csv').iloc[:,1:]
y=data.iloc[:,-1]
feature_names=data.columns[:-1]
print('0'*100)
ent_obj=EntropyUtils()
for feat in feature_names:
    print(feat,":",ent_obj.info_gain(data.loc[:,feat],y))
