# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/28 22:07
@file: example04.py
@desc: 
"""
import numpy

world_alcohol = numpy.genfromtxt("./world_alcohol.txt", delimiter=',', dtype=str, skip_header=1)
print(world_alcohol)
uruguay_other_1986 = world_alcohol[1, 4]
print(uruguay_other_1986)
third_country = world_alcohol[2, 2]
print(uruguay_other_1986)
print(third_country)

vector = numpy.array([5, 10, 15, 20])
print(vector[0:3])

matrix = numpy.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45]
])

print(matrix[:, 1])

print(matrix[:,0:2])
print(matrix[1:3,0:2])
vector=numpy.array([5,10,15,20])
equal_to_len=vector==10
print(vector[equal_to_len])

numpy.array([
    [5,10,15],
    [20,25,30],
    [35,40,45],
])

second_column_25=(matrix[:,1]==25)
print(second_column_25)
print(matrix[second_column_25, :])
