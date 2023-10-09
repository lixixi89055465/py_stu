# -*- coding: utf-8 -*-
# @Time    : 2023/9/2 10:24
# @Author  : nanji
# @Site    : 
# @File    : testplt_step.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(14)
y = np.sin(x / 2)
plt.figure(121)
plt.step(x, y + 2, where='pre', color='r')
plt.plot(x, y + 2, 'o--', color='grey')

plt.step(x, y + 1, where='mid', label='mid')
plt.plot(x, y + 1, 'o--', color='grey', alpha=0.3)

plt.step(x, y , where='post', label='post')
plt.plot(x, y , 'o--', color='grey', alpha=0.3)


plt.subplot(122)
plt.plot(x, y + 2, drawstyle='steps', label='steps (=steps-pre)')
plt.plot(x, y + 2, 'o--', color='grey', alpha=0.3)

plt.plot(x, y + 1, drawstyle='steps-mid', label='steps-mid')
plt.plot(x, y + 1, 'o--', color='grey', alpha=0.3)

plt.plot(x, y, drawstyle='steps-post', label='steps-post')
plt.plot(x, y, 'o--', color='grey', alpha=0.3)
plt.show()
