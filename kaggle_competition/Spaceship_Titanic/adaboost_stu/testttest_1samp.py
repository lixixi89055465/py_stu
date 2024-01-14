# -*- coding: utf-8 -*-
# @Time : 2024/1/12 23:36
# @Author : nanji
# @Site : 
# @File : testttest_1samp.py
# @Software: PyCharm 
# @Comment :
# Import numpy and scipy
# Import numpy and scipy
import numpy as np
from scipy import stats

# Create array of worker bottling rates between 10 and 20 bottles/min
pre_training = np.random.randint(low=10, high=20, size=30)

# Define "training" function and apply
def apply_training(worker):
    return worker + np.random.randint(-1, 4)

post_training = list(map(apply_training, pre_training))

# Run a paired t-test to compare worker productivity before & after the training
tstat, pval = stats.ttest_rel(post_training, pre_training)

# Display results
print("t-stat: {:.2f}   pval: {:.4f}".format(tstat, pval))
