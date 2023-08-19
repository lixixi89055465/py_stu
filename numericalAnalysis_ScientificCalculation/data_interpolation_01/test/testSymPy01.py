# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/8/19 21:52
# @Author  : nanji
# @File    : testSymPy01.py
# @Description :  机器学习 https://zhuanlan.zhihu.com/p/111573239

from sympy import *

x, y, z = symbols('x y z')
y = expand((x + 1) ** 2)  # expand() 是展开函数
print(y)

z = Rational(1, 2)  # 构造分数 1/2
print('0' * 100)
print(z)
x = symbols('x')
expr = cos(x) + 1
print('1' * 100)
print(expr)
print('2' * 100)
# 采用符号变量的 subs 方法进行替换操作，例如：
print(expr.subs(x, 0))
x=symbols('x')
str_expr='x**2 + 2*x +1'
expr=sympify(str_expr)
print('3'*100)
print(expr)
print('4'*100)
print(pi.evalf(3))
print('5'*100)

