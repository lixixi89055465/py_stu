# -*- coding: utf-8 -*-
# @Time    : 2023/10/9 下午9:14
# @Author  : nanji
# @Site    : 
# @File    : test_nb_classifier.py
# @Software: PyCharm 
# @Comment :

import pandas as pd
import numpy as np
from machinelearn.bayesian_07.lecture.naive_bayes_classifier import NaiveBayesClassifier

wm = pd.read_csv("../datasets/watermelon.csv").dropna()
X, y = np.asarray(wm.iloc[:, 1:-1]), np.asarray(wm.iloc[:, -1])
nbc = NaiveBayesClassifier(is_binned=True, feature_R_idx=[6, 7])
# nbc = NaiveBayesClassifier(is_binned=False, feature_R_idx=[6, 7])
nbc.fit(X, y)
