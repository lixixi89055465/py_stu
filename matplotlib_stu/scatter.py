# coding=utf-8

from matplotlib import pyplot as plt
from matplotlib import font_manager
import random

my_font = font_manager.FontProperties(fname="/System/Library/Fonts/PingFang.ttc")

y_3 = [11, 17, 16, 18, 20, 17, 16, 18, 13, 10, 20, 13, 12, 11, 20, 18, 10, 10, 10, 15, 18, 18, 13, 20, 17, 14, 10, 17,
       16, 10, 13]
y_10 = [17, 19, 13, 19, 16, 12, 11, 15, 14, 12, 10, 12, 13, 14, 19, 17, 17, 16, 15, 20, 20, 18, 18, 19, 13, 11, 15, 14,
        16, 11]
x_3 = range(1, 32)
x_10 = range(51, 81)

# 设置图形大小
plt.figure(figsize=(20, 8), dpi=80)
# 使用scatter 方法绘制散点图，和之前绘制折线图的唯一区别
plt.scatter(x_3, y_3, label="3月份")
plt.scatter(x_10, y_10, label="10月份")

_x = list(x_3) + list(x_10)
_xtick_labels = ["3月{}日".format(i) for i in x_3]
_xtick_labels += ["10月{}日".format(i - 50) for i in x_10]

# 调整 x 轴的刻度
plt.xticks(_x[::3], _xtick_labels[::3], fontproperties=my_font, rotation=45)

# 添加图例
plt.legend(loc="upper left", prop=my_font)
# 添加描述信息
plt.xlabel("时间 ", fontproperties=my_font)
plt.ylabel("温度 ", fontproperties=my_font)
plt.title("标题", fontproperties=my_font)
# 展示
plt.show()
