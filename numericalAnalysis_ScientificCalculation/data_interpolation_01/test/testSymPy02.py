# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/8/19 22:05
# @Author  : nanji
# @File    : testSymPy02.py.py
# @Description :  机器学习 https://space.bilibili.com/512662380

import numpy
from sympy import *

a = numpy.pi / 3
# print(a)
x = symbols('x')
expr = sin(x)
f = lambdify(x, expr, 'numpy')
print(f(a))
print('0' * 100)
print(expr)
print('1' * 100)
b = expr.subs(x, pi / 3)
print(b)
print('2' * 100)
c = simplify(sin(x) ** 2 + cos(x) ** 2)
print(c)
alpha_mu = symbols('alpha_mu')
d = simplify(2 * sin(alpha_mu) * cos(alpha_mu))
print('3' * 100)
print(d)
x_1 = symbols('x_1')
print('4' * 100)
e = expand((x_1 + 1) ** 2)
print(e)
f = factor(x ** 3 - x ** 2 + x - 1)
print('5' * 100)
print(f)
from sympy.abc import x, y, z

expr = x * y + x - 3 + 2 * x ** 2 - z * x ** 2 + x ** 3
g = collect(expr, x)  # 利用 collect 合并同类项，例如：
print('6' * 100)
print(g)
h = cancel((x ** 2 + 2 * x + 1) / (x ** 2 + x))  # 消去分子分母的公因式使用 cancel 函数，
print('7' * 100)
print(h)
expr = (4 * x ** 3 + 21 * x ** 2 + 10 * x + 12) / (x ** 4 + 5 * x ** 3 + 5 * x ** 2 + 4 * x)
print('8' * 100)
print(expr)
i = apart(expr)
print(i)
