# coding=utf-8
import numpy as np

us_file_path = "./youtube_video_data/US_video_data_numbers.csv"
uk_file_path = "./youtube_video_data/GB_video_data_numbers.csv"

t1 = np.loadtxt(us_file_path, delimiter=",", dtype=np.int, unpack=True)
t2 = np.loadtxt(us_file_path, delimiter=",", dtype=np.int, unpack=False)
print(t1)
print("*" * 100)
print(t2)
print("-" * 100)
print(t2[2])
# 取连续的多行
print(t2[2:])

# 取不连续的多行
print(t2[[2, 3, 10]])
print("1" * 100)
# 取列
print(t2[1, :])
print(t2[2, :])
print("2" * 100)

print(t2[[2, 2, 3], :])

# 取不连续的多列
print(t2[:, [0, 1, 2, 3]])
a = t2[2, 3]
# 去多行和多列，取第三行，第四列的值
print(a)
print(type(a))
# 取多行和多列，取第三行到第五行，第四列的值
print("2" * 100, "\n")
print(t2[2:5, 1:4])
# 取多个不相邻的点
print("3" * 100)
print(t2[0:6, ])
c = t2[[0, 2, 2], [0, 1, 3]]
print(c)
# 布尔索引
t2 = np.arange(24).reshape(4, 6)
print(t2 < 10)
t2[t2 < 10] = 3
print(t2)
print(t2[t2 > 20])
# 三目运算
print(np.where(t2 < 10, 0, 10))
t2 = t2.astype(float)
# nan 是float 型，不能直接复制给int 型数值
t2[3, 4] = np.nan
print(t2)
print(np.isnan(t2))

# t2[np.isnan(t2)]=0
# print(t2)
print(np.sum(t2))
np.sum(t2)
t3=np.arange(0,12).reshape(3,4)
print(t3)
print(np.sum(t3, axis=0))
print(np.sum(t3, axis=1))
print(t2)
print(t2.sum(axis=0))
print(t2.mean(axis=0))
print(np.median(t2))
print(t2.max(axis=0))
print(t2.min(axis=0))
print(np.ptp(t2))
np.ptp(t2,axis=0)
np.ptp(t2,axis=1)

print(t2.std(axis=0))