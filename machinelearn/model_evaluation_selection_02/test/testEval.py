# -*- coding: utf-8 -*-
# @Time    : 2023/10/9 15:25
# @Author  : nanji
# @Site    : 
# @File    : testEval.py
# @Software: PyCharm 
# @Comment :
x = 7
print(eval("2*x"))
print('0' * 100)
print(eval('pow(2,2)'))
a = "[[1,2], [3,4], [5,6], [7,8], [9,0]]"
print(type(a))
b = eval(a)
print(type(b))
print(b)
print('1' * 100)
a = "{1: 'a', 2: 'b'}"
print(type(a))
b = eval(a)
print(type(b))
print(b)

print('2' * 100)
a = "{1: 'a', 2: 'b'}"
print(type(a))
b = eval(a)
print(type(b))
print(b)
print('3'*100)

a = "([1,2], [3,4], [5,6], [7,8], (9,0))"
print(type(a))
b=eval(a)
print(type(b))
print(b)
