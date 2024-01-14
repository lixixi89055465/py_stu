# -*- coding: utf-8 -*-
# @Time : 2024/1/13 14:15
# @Author : nanji
# @Site : 
# @File : testppf.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import scipy.stats as stats
p = 0.025
weight = [122, 130, 139, 168, 125, 160, 155, 189, 107, 164]
low=np.mean(weight)-stats.norm.ppf(q=1-p)*(np.sqrt(1.2)/np.sqrt(len(weight)))
high=np.mean(weight)+stats.norm.ppf(q=1-p)*(np.sqrt(1.2)/np.sqrt(len(weight)))

