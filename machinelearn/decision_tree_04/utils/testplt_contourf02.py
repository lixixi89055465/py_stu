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
x=np.linspace(-3,3,50)# 生成连续数据
y=np.linspace(-3,3,50)# 生成连续数据
X,Y=np.meshgrid(x,y)
z=X**2+Y**2
# c=plt.contour(x,y,z,[2,5,8,10])# 画等高线#使用
c=plt.contourf(x,y,z,[2,5,8,10])# 画等高线#使用
plt.show()
