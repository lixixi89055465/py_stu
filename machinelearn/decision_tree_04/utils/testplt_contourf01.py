# -*- coding: utf-8 -*-
# @Time    : 2023/9/30 21:36
# @Author  : nanji
# @Site    : 
# @File    : testplt_contourf.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 计算x,y坐标对应的高度值
def f(x, y):
    return (1 - x / 2 + x ** 2 + y ** 5) * np.exp(-x ** 2 - y ** 2)


# 生成x,y的散度
n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
# 把x,y数据生成mesh网络状的数据，因为等高线的显示实在网络的基础上添加上高度值
X, Y = np.meshgrid(x, y)

# 填充等高线
plt.contourf(X, Y, f(X, Y),cmap=plt.cm.hot)
# 显示图表
plt.show()
