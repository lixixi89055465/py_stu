# -*- coding: utf-8 -*-
# @Time : 2024/1/3 22:02
# @Author : nanji
# @Site : https://www.kaggle.com/code/samuelcortinhas/spaceship-titanic-a-complete-guide
# @File : exam01.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

sns.set(style='darkgrid', font_scale=1.4)
from imblearn.over_sampling import SMOTE
