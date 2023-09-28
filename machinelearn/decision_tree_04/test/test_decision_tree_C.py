# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/9/29 0:55
# @Author  : nanji
# @File    : test_decision_tree_C.py
# @Description :
import numpy as np
from machinelearn.decision_tree_04.decision_tree_C import DecisionTreeClassifier
import pandas as pd
data=pd.read_csv('../../data/watermelon.csv').iloc[:,1:]
X,y=data.iloc[:,:-1],data.iloc[:,-1:]
dtc=DecisionTreeClassifier(dbw_feature_idx=[6,7])
dtc.fit(X,y)


