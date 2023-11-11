# -*- coding: utf-8 -*-
# @Time    : 2023/11/11 21:01
# @Author  : nanji
# @Site    : 
# @File    : test01.py
# @Software: PyCharm 
# @Comment :https://www.bilibili.com/video/BV1CV411Y7i4/?p=2&spm_id_from=pageDriver&vd_source=50305204d8a1be81f31d861b12d4d5cf

import os
import torch

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X)
print('1' * 100)
print((a * X).shape)
