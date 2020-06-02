# coding=utf-8
import numpy as np

us_file_path = "./youtube_video_data/"
uk_file_path = "./youtube_video_data/"

t1 = np.loadtxt(us_file_path, delimiter=",", dtype=np.int, unpack=True)
t2 = np.loadtxt(us_file_path, delimiter=",", dtype=np.int, unpack=False)
print(t1)
print("*" * 100)
print(t2)
