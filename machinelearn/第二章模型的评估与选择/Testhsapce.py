import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['axes.unicode_minus'] = False

x = np.linspace(0, 2 * np.pi, 500)
y1 = np.sin(x) * np.cos(x)
y2 = np.exp(-x)
y3 = np.sqrt(x)
y4 = x / 4

fig, ax = plt.subplots(4, 1, facecolor='beige', sharex=True,
                       subplot_kw=dict(facecolor='seashell'))

fig.subplots_adjust(left=0.9, right=0.98, bottom=0.05,
                    top=0.95, hspace=0)

ax[0].plot(x, y1, c='r', lw=2)
ax[1].plot(x, y2, c='y', ls="--")
ax[2].plot(x, y3, c='g', ls=":")
ax[3].plot(x, y4, c='m', ls='-.', lw=2)

plt.show()
