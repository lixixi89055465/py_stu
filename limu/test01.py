# -*- coding: utf-8 -*-
# @Time    : 2023/11/11 21:01
# @Author  : nanji
# @Site    : 
# @File    : 48_2.py
# @Software: PyCharm 
# @Comment :https://www.bilibili.com/video/BV1CV411Y7i4/?p=2&spm_id_from=pageDriver&vd_source=50305204d8a1be81f31d861b12d4d5cf

import os
import torch

print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.__version__)