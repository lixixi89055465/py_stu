# -*- coding: utf-8 -*-
# @Time    : 2023/10/29 20:19
# @Author  : nanji
# @Site    : 
# @File    : Test03.py
# @Software: PyCharm 
# @Comment :
import matplotlib.pyplot as plt
import numpy as np


def f(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


x = np.linspace(-7, 7, 50)
y = np.linspace(-7, 7, 50)

z = np.zeros((50, 50))
for i, a in enumerate(x):
    for j, b in enumerate(y):
        z[i, j] = f(a, b)
xx, yy = np.meshgrid(x, y)

fig, ax = plt.subplots()
c = ax.pcolormesh(xx, yy, z.T,shading='auto' ,cmap='jet')
fig.colorbar(c, ax=ax)
plt.show()
