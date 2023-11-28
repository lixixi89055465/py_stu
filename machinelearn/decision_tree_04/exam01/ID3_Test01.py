import sys
import os

CURRENT_DIR = os.path.split(os.path.abspath("."))[0]  # 当前目录
config_path1 = CURRENT_DIR.rsplit('/', 0)[0]  # 上1级目录
config_path2 = CURRENT_DIR.rsplit('/', 1)[0]  # 上2级目录
config_path3 = CURRENT_DIR.rsplit('/', 2)[0]  # 上3级目录
sys.path.append(config_path1)
sys.path.append(config_path2)
sys.path.append(config_path3)
print(config_path1)
print(config_path2)
print(config_path3)
print('1'*100)

import random
import numpy as np
import matplotlib.pyplot as plt
from machinelearn.decision_tree_04.entropy_utils import EntropyUtils

# 看特征的取值的个数对信息增益的影响
epochs = 100
class_num_x = 100
class_num_y = 2
num_samples = 100
info_gains, info_gain_rates = [], []  # 信息增益ID3,信息增益率 C4.5
for _ in range(0, epochs):
    info_gain_, info_gain_rate_ = [], []
    for class_x in range(2, class_num_x):
        x, y = [], []
        for _ in range(num_samples):
            x.append(random.randint(1, class_x))
            y.append(random.randint(1, class_num_y))
        info_gain_.append(EntropyUtils().info_gain(x, y))  # 信息增益 ID3
        info_gain_rate_.append(EntropyUtils().info_gain_rate(x, y))  # 信息增益率 C4.5
    info_gains.append(info_gain_)
    info_gain_rates.append(info_gain_rate_)
# 信息增益和信息增益率可视化

plt.figure(figsize=(8, 6))
plt.plot(np.asarray(info_gains).mean(axis=0), lw=1.5, label='Information Gain')
plt.plot(np.asarray(info_gain_rates).mean(axis=0), lw=1.5, label='Information Gain Rate')
plt.xlabel('The number of Values (Feature - X)', fontsize=12)
plt.ylabel('information gain / information gain rate ', fontsize=12)
plt.title('Information gain VS information Gain rate increases with \n'
          'the number of sample characteristics', fontsize=14)
plt.grid(ls=':')
plt.legend(frameon=False)
plt.savefig('a.png')
plt.show()
print('exit ')
