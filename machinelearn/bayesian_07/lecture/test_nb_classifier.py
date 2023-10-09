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
nbc = NaiveBayesClassifier()
nbc.fit(X,y)
