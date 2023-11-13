# -*- coding: utf-8 -*-
# @Time    : 2023/11/11 19:41
# @Author  : nanji
# @Site    : 
# @File    : pytorch安装.py
# @Software: PyCharm 
# @Comment :https://www.bilibili.com/video/BV1s3411F7c5/?p=4&spm_id_from=pageDriver&vd_source=50305204d8a1be81f31d861b12d4d5cf

import torch

print(torch.__version__)
x = torch.randn(3, 4)
x.requires_grad=True
print(x)
b = torch.randn(3, 4, requires_grad=True)
t = x + b
y = t.sum()
y.backward()
print(b.grad)
print(x.requires_grad, t.requires_grad, b.requires_grad)
