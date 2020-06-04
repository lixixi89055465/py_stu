# coding=utf-8
import numpy as np

us_data = "./youtube_video_data/US_video_data_numbers.csv"
uk_data = "./youtube_video_data/GB_video_data_numbers.csv"

# 记载国家数据
us_data = np.loadtxt(us_data, delimiter=",", dtype=int)
uk_data = np.loadtxt(uk_data, delimiter=",", dtype=int)

# 添加国家信息
# 构造全为0的数据
np.zeros((us_data.shape[0], 1)).astype(int)
ones_data = np.ones(uk_data.shape[0], 1).astype(int)

# 拼接两组数据
final_data = np.vstack((us_data, uk_data))
print(final_data)
