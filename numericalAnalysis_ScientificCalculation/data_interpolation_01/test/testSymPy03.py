# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/8/19 23:13
# @Author  : nanji
# @File    : testSymPy03.py.py
# @Description :  机器学习 https://zhuanlan.zhihu.com/p/111573239 微积分符号计算
from sympy import *

x = sympify('x')
a = diff(cos(x), x)
print(a)
b = diff(x ** 4, x, 3)
print(b)
print('1' * 100)
expr = cos(x)
c = expr.diff(x, 2)
print(c)
from sympy.abc import x, y, z

expr = exp(x * y * z)
d = diff(expr, x)
print(d)

e = integrate(cos(x), x)
print('2' * 100)
print(e)
f = integrate(exp(-x), (x, 0, oo))
print('3' * 100)
print(f)

g = integrate(exp(-x ** 2 - y ** 2), (x, -oo, oo), (y, -oo, oo))
print(g)
print('4' * 100)
h = limit(sin(x) / x, x, 0)
print(h)
i=limit(1/x,x,0,'+')
print('5'*100)
print(i)
print('6'*100)
expr=sin(x)
j=expr.series(x,0,4)
print(j)
print(expr.subs(x, 10))



