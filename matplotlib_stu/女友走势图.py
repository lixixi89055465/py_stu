from matplotlib import pyplot as plt
import matplotlib

from matplotlib import font_manager
import random

my_font = font_manager.FontProperties(fname="/System/Library/Fonts/PingFang.ttc")
y = [1, 0, 1, 1, 2, 4, 3, 2, 3, 4, 4, 5, 6, 5, 4, 3, 3, 1, 1, 1]
x = range(11, 31)

plt.figure(figsize=(20, 8), dpi=80)
plt.plot(x, y)
# 跳着刻度
x_tick_labels = ["年龄{}岁".format(x[i]) for i in range(20)]
# 取步长，数字和字符串一一对应，数据的长度一样
plt.xticks(list(x), list(x_tick_labels), rotation=45, fontproperties=my_font)

y_tick_labels = ["女友{}个".format(y[i]) for i in range(20)]
plt.yticks(list(y), list(y_tick_labels), rotation=45, fontproperties=my_font)

plt.xlabel("女友个数", fontproperties=my_font)
plt.ylabel("年龄", fontproperties=my_font)

plt.grid(alpha=0.4)
plt.title("年龄和女友的关系", fontproperties=my_font)
plt.show()
