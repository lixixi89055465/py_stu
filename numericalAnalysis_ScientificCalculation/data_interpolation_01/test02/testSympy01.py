# -*- coding: utf-8 -*-
# @Time    : 2023/9/4 15:50
# @Author  : nanji
# @Site    : 
# @File    : testSympy01.py
# @Software: PyCharm 
# @Comment :  https://zhuanlan.zhihu.com/p/111573239
import math

print(math.pi)
print(math.sin(math.pi))

import sympy

print('0' * 100)
print(sympy.sin(sympy.pi))
x, y = sympy.symbols('x y')

print('1' * 100)
vars = sympy.symbols('x_1:5')
print(vars)
x, y, z = sympy.symbols('x y z')
y = sympy.expand((x + 1) ** 2)
print('2' * 100)
print(y)
z = sympy.Rational(1, 2)
print('3' * 100)
print(z)
print('4' * 100)
x = sympy.symbols('x')
expr = sympy.cos(x) + 1
print(expr)
print('5' * 100)
str_exp = 'x**2+2*x+1'
expr = sympy.sympify(str_exp)
print(expr)

print('6'*100)
print(sympy.pi.evalf(3))
