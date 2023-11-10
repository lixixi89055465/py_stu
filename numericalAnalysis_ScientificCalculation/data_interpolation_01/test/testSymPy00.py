# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/8/19 21:39
# @Author  : nanji
# @File    : testSymPy00.py
# @Description :  机器学习 https://space.bilibili.com/512662380
import math

import sympy
from sympy import *

print(math.sqrt(2))
print(sympy.sqrt(2))

x, y = symbols('x y')
print('0' * 100)
print(x, y)

from sympy.abc import x, y

x = symbols('x', positive=True)
vars = symbols('x_1:5')
print('1'*100)
print(vars)