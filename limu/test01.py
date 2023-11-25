# -*- coding: utf-8 -*-
# @Time    : 2023/11/11 21:01
# @Author  : nanji
# @Site    : 
# @File    : test01.py
# @Software: PyCharm 
# @Comment :https://www.bilibili.com/video/BV1CV411Y7i4/?p=2&spm_id_from=pageDriver&vd_source=50305204d8a1be81f31d861b12d4d5cf

import os
import torch

X = torch.ones((3, 2))

dropout = 0.5
print((torch.randn(X.shape) > dropout).float())
