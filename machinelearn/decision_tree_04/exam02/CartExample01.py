import numpy as np

y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])  # y的值


class TreeNode(object):
	def __init__(self, tempR, tempC):
		self.R = tempR
		self.C = tempC
		self.left = None
		self.right = None


def cart(start, end):
	if end - start >= 1:
		result = []
		for s in range(start + 1, end + 1):
			y1 = y[start:s]
			y2 = y[s: end + 1]
			result.append((y1.std() ** 2) * len(y1) + (y2.std() ** 2) * len(y2))
		index1 = result.index(min(result)) + start
		root = TreeNode(y[start:end + 1], min(result))
		print('节点元素为', y[start:end + 1], '\n s = ', index1 + 1, \
			  ',最小平方误差为', min(result))
		root.left = cart(start, index1)
		root.right = cart(index1 + 1, end)
	else:
		root = None
	return root

if __name__ == '__main__':
	root=cart(0,9)
