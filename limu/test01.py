# -*- coding: utf-8 -*-
# @Time    : 2023/11/11 21:01
# @Author  : nanji
# @Site    : 
# @File    : test01.py
# @Software: PyCharm 
# @Comment :https://www.bilibili.com/video/BV1CV411Y7i4/?p=2&spm_id_from=pageDriver&vd_source=50305204d8a1be81f31d861b12d4d5cf

import os
import torch


def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad == d / a)
print(a.grad)
