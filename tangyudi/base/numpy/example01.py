# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 21:50
@file: example01.py
@desc: 
"""
import numpy
world_alcohol=numpy.genfromtxt('world_alcohol.txt',delimiter=',',dtype=str)
print(type(world_alcohol))
print(world_alcohol)
print(help(numpy.genfromtxt))