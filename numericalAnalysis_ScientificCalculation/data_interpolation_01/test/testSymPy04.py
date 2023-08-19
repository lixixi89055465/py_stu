# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/8/19 23:30
# @Author  : nanji
# @File    : testSymPy04.py
# @Description :  https://zhuanlan.zhihu.com/p/111573239 解方程 使用 solveset 求解方程。
import sympy
from sympy import *
from sympy.abc import *

a = solveset(Eq(x ** 2 - x, 0), x, domain=sympy.Reals)
print(a)

f = symbols('f', cls=Function)
diffeq = Eq(f(x).diff(x, 2) - 2 * f(x).diff(x) + f(x), sin(x))
print('0' * 100)
print(diffeq)
a = Matrix([[1, -1], [3, 4], [0, 2]])
print(a)
b = Matrix([1, 2, 3])
print(b)
c = Matrix([[1], [2], [3]]).T
print(c)
print('1' * 100)
print(eye(4))
print(zeros(4))
print('2' * 100)
print(ones(4))
print(diag(1, 2, 3, 4))
a = Matrix([[1, -1], [3, 4], [0, 2]])
print('3' * 100)
print(a)
print(a.T)
print('4' * 100)
M = Matrix([[1, 3], [-2, 3]])
print(M ** 2)
print('5'*100)
print(M ** -1)
M = Matrix([[1, 0, 1], [2, -1, 3], [4, 3, 2]])
print('6'*100)
print(M)
print(M.det())
print('7'*100)
M = Matrix([[3, -2,  4, -2], [5,  3, -3, -2], [5, -2,  2, -2], [5, -2, -3,  3]])
print(M)
print(M.eigenvals())

