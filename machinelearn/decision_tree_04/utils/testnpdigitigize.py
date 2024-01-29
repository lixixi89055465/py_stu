# -*- coding: utf-8 -*-
# @Time : 2024/1/29 16:41
# @Author : nanji
# @Site : 
# @File : testnpdigitigize.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l
import numpy as np

import random


import numpy as np
bins = np.array(range(-99, 102, 3))
print(bins)
print('0'*100)
a = np.digitize(-98,bins) #a=1
b = np.digitize(68,bins)  #b=56
print(a)
print('1'*100)
print(b)