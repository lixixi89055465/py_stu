# coding=utf-8

from matplotlib import pyplot as plt

import random

x = range(0, 120)
y = [random.randint(20, 35) for i in range(120)]

plt.figure(figsize=(20, 8), dpi=80)
plt.plot(x, y)
# 调整x轴的刻度
_x = list(x)[::10]
# _xtick_labels = ["hello,{}".format(i) for i in _x]
_xtick_labels = ["11点{}分".format(i) for i in range(60)]

plt.xticks(_x[::3], _xtick_labels[::3])

plt.show()
