# -*- coding: utf-8 -*-
# @Time    : 2023/9/4 16:09
# @Author  : nanji
# @Site    : 
# @File    : testSympy02.py
# @Software: PyCharm 
# @Comment :  https://zhuanlan.zhihu.com/p/111573239#:~:text=lambdify%20%E5%87%BD%E6%95%B0%E7%9A%84%E5%8A%9F%E8%83%BD%E5%B0%B1%E6%98%AF%E5%8F%AF%E4%BB%A5%E5%B0%86%20SymPy%20%E8%A1%A8%E8%BE%BE%E5%BC%8F%E8%BD%AC%E6%8D%A2%E4%B8%BA%20NumPy%20%E5%8F%AF%E4%BB%A5%E4%BD%BF%E7%94%A8%E7%9A%84%E5%87%BD%E6%95%B0%EF%BC%8C%E7%84%B6%E5%90%8E%E7%94%A8%E6%88%B7%E5%8F%AF%E4%BB%A5%E5%88%A9%E7%94%A8%20NumPy%20%E8%AE%A1%E7%AE%97%E8%8E%B7%E5%BE%97%E6%9B%B4%E9%AB%98%E7%9A%84%E7%B2%BE%E5%BA%A6%E3%80%82
import sympy
import numpy

a = numpy.pi / 3
x = sympy.symbols('x')
expr = sympy.sin(x)
print('0' * 100)
print(expr)
f = sympy.lambdify(x, expr, 'numpy')
print(f(a))
print('1' * 100)
print(expr.subs(x, sympy.pi / 3))

from sympy import sin, cos

print('2' * 100)
print(sympy.simplify(sin(x) ** 2 + cos(x) ** 2))

print('3' * 100)
alpha_mu = sympy.symbols('alpha_mu')
print(sympy.simplify(2 * sin(alpha_mu) * cos(alpha_mu)))

print('4' * 100)
x_1 = sympy.symbols('x_1')
print(sympy.expand((x_1 + 1) ** 2))
print('5' * 100)
print(sympy.factor(x ** 3 - x ** 2 + x - 1))

from sympy.abc import x, y, z

expr = x * y + x - 3 + 2 * x ** 2 - z * x ** 2 + x ** 3
print('6' * 100)
print(sympy.collect(expr, x))
print('7' * 100)
print(sympy.diff(cos(x), x))
print('8' * 100)
print(sympy.diff(x ** 4, x, 3))
print('9'*100)
expr = sympy.exp(x*y*z)
print(sympy.diff(expr, x))
# 求不定积分
print('0'*100)
print(sympy.integrate(cos(x), x))