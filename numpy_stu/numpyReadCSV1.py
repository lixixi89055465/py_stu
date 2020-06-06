# coding: utf-8
# @Time    : 2020/6/6 9:17 AM
# @Author  : lixiang
# @File    : numpyReadCSV1.py

import numpy as np
from matplotlib import pyplot as plt

us_file_path = "./youtube_video_data/US_video_data_numbers.csv"
uk_file_path = "./youtube_video_data/GB_video_data_numbers.csv"

t_us = np.loadtxt(us_file_path, delimiter=",", dtype="int")
print(t_us)
t_us_comments = t_us[:, -1]
print(t_us_comments.max(), t_us_comments.min())
# t_us_comments=t_us_comments[t_us_comments<5000]
d = 1000
bin_nums = (t_us_comments.max() - t_us_comments.min()) // d
print(bin_nums)
plt.hist(t_us_comments, bin_nums)
# 绘图
plt.figure(figsize=(20, 8), dpi=80)
plt.show()
