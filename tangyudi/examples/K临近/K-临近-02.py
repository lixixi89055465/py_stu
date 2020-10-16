# coding: utf-8
# @Time    : 2020-07-22 22:35
# @Author  : lixiang
# @File    : K-临近-01.py




import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

img=plt.imread('/Users/lixiang/PycharmProjects/py3/py_stu/tangyudi/examples/K临近/0_2.png')
plt.imshow(img)

print(img.shape)