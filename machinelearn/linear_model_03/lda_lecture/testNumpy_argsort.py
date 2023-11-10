# -*- coding: utf-8 -*-
# @Time    : 2023/9/23 下午5:11
# @Author  : nanji
# @Site    : 
# @File    : testNumpy_argsort.py
# @Software: PyCharm 
# @Comment :
import numpy as np
# 一维数组
x = np.array([3, 1, 2])
print('一维数组的排序结果：{}'.format(np.argsort(x))[::-1])
# 二维数组
x = np.array([[0, 3], [2, 2]])
print('被排序的数组为：\n{}'.format(x))
# 沿着列方向进行排序
ind = np.argsort(x, axis=0)
print('列方向的排序索引为：\n{}'.format(ind))
print('列方向的排序结果为：\n{}'.format(np.take_along_axis(x, ind, axis=0)))
# 沿着行方向进行排序
ind = np.argsort(x, axis=1)
print('行方向的排序索引为：\n{}'.format(ind))
print('行方向的排序结果为：\n{}'.format(np.take_along_axis(x, ind, axis=1)))
# n维数组元素排序后的索引
ind = np.unravel_index(np.argsort(x, axis=None), x.shape)
print('多维数组拉伸为一维后排序的索引为：{}'.format(ind))
print('将多维数组拉伸为一维后进行排序：{}'.format(x[ind]))
# 根据指定的键进行排序
x = np.array([(1, 0), (0, 1)], dtype=[('x', '<i4'), ('y', '<i4')])
print('原始数组为：')
x
# 沿着指定顺序进行排序
print('先对比x字段，再对比y字段：{}'.format(np.argsort(x, order=('x','y'))))
print('先对比y字段，再对比x字段：{}'.format(np.argsort(x, order=('y','x'))))