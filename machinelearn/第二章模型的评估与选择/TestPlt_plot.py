import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
line_style = ['-', '--', '-.', ':']
dic1 = [[0, 1, 2], [3, 4, 5]]
x = pd.DataFrame(dic1)
dic2 = [[2, 3, 2], [3, 4, 3], [4, 5, 4], [5, 6, 5]]
y = pd.DataFrame(dic2)
# 循环输出所有"颜色"与"线型"
for i in range(2):
    for j in range(4):
        plt.plot(x.loc[i], y.loc[j], color[i * 4 + j] + line_style[j])
plt.show()
