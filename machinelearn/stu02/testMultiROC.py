# -*- coding: utf-8 -*-
# @Time    : 2023/10/7 17:23
# @Author  : nanji
# @Site    : 
# @File    : testBinaryROC.py
# @Software: PyCharm 
# @Comment :  3. 性能度量——ROC与AUC+二分类
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import random
