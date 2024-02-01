# -*- coding: utf-8 -*-
# @Time : 2024/2/1 15:17
# @Author : nanji
# @Site : 
# @File : test_entropy.py
# @Software: PyCharm 
# @Comment :
import numpy as np

y = np.array([5.56, 5.7, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])


class TreeNode(object):
	def __init__(self, tempR, tempC):
		self.R = tempR
		self.c = tempC
		self.left = None
		self.right = None


def CART(start, end):
	if end - start >= 1:
		result = []
		for s in range(start + 1, end + 1):
			y1 = y[start:s]
			y2 = y[s, end + 1]
			result.append((y1.std() ** 2) * y1.size + (y2.std() ** 2) * y2.size)
		index1 = result.index(min(result)) + start
		root = TreeNode(y[start:end + 1], min(result))
		root.left = CART(start, index1)
		root.right = CART(index1 + 1, end)
	else:
		root = None
	return root


if __name__ == '__main__':
	root = CART(0, 9)
