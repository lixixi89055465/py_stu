# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/28 22:34
@file: example05.py
@desc: 
"""
import numpy
vector=numpy.array([5,10,15,20])
equal_to_ten_and_five=(vector==10)&(vector==5)
print(equal_to_ten_and_five)
