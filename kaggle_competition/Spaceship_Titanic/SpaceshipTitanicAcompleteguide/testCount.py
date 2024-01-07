# -*- coding: utf-8 -*-
# @Time : 2024/1/6 9:48
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/51003123
# @File : testCount.py
# @Software: PyCharm 
# @Comment :
import itertools

# for i in itertools.count(10, 2):
# 	print(i)
# 	if i > 20:
# 		break
print('0' * 100)
# for i in itertools.cycle('abcd'):
# 	print(i)


# for i in itertools.repeat('abcd', 5):
# 	print(i)

# from itertools import chain
# list1=[1,2,3]
# tuple1=('a','b','c')
# for element in chain(list1,tuple1):
# 	print(element)


# list1=[1,2]
# tuple1=('a','b','c')
# for pair in itertools.zip_longest(list1,tuple1,fillvalue=100):
# 	print(pair)


# my_list = [1, 2, 3, 4]
# for combo in itertools.combinations(my_list, 3):
# 	print(combo)

# for perm in itertools.permutations(my_list,2):
# 	print(perm)

# for i in itertools.product([1, 2, 3], [4, 5, 6]):
# 	print(i)
# for i in itertools.product([1, 2, 3], repeat=3):
# 	print(i)
# for i in itertools.permutations('abc'):
# 	print(i)

for i in itertools.combinations('abc', 2):
	print(i)
