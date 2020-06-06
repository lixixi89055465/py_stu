# coding: utf-8
# @Time    : 2020/6/6 9:17 AM
# @Author  : lixiang
# @File    : numpyReadCSV1.py

import numpy as np
from matplotlib import pyplot as plt

us_file_path = "./youtube_video_data/US_video_data_numbers.csv"
uk_file_path = "./youtube_video_data/GB_video_data_numbers.csv"

t_us = np.loadtxt(us_file_path, delimiter=",", dtype="int")
t_uk = np.loadtxt(uk_file_path, delimiter=",", dtype="int")
t_uk=t_uk[t_uk[:,1]<500000]
t_uk_comment=t_uk[:,-1]
t_uk_like=t_uk[:,1]

plt.figure(figsize=(20,8),dpi=80)
plt.scatter(t_uk_like,t_uk_comment)
plt.show()
