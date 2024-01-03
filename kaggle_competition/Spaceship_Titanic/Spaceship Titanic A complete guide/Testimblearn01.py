# -*- coding: utf-8 -*-
# @Time : 2024/1/3 22:10
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/95020088
# @File : Testimblearn01.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l
from imblearn.under_sampling import RandomUnderSampler
rus=RandomUnderSampler(random_state=0)
