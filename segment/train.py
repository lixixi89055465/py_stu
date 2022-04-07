# 初始化
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from skimage import io

# from __future__ import print_function

# %matplotlib内联
plt.rcParams['figure.figsize'] = (15.0, 12.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# 自动重载外部模块
# %load_ext autoreload
# %autoreload 2
# 为聚类生成随机数据点

# 采用seed保证生成结果的一致
np.random.seed(0)

# 载入并显示图像
img = io.imread('train.jpg')
H, W, C = img.shape

plt.imshow(img)
plt.axis('off')
plt.show()

from segmentation import color_features
np.random.seed(0)
features=color_features(img)
# 结果检测
# assert features.shape == (H * W, C),\
#     "Incorrect shape! Check your implementation."
#
# assert features.dtype == np.float,\
#     "dtype of color_features should be float."

from segmentation import kmeans_fast
assignments=kmeans_fast(features,8)
segments = assignments.reshape((H, W))

# 展示图像分割结果
plt.imshow(segments, cmap='viridis')
plt.axis('off')
plt.show()



from segmentation import color_position_features
np.random.seed(0)

features = color_position_features(img)

# 结果检测
assert features.shape == (H * W, C + 2),\
    "Incorrect shape! Check your implementation."

assert features.dtype == np.float,\
    "dtype of color_features should be float."

assignments = kmeans_fast(features, 8)
segments = assignments.reshape((H, W))

# 图像分割结果显示
plt.imshow(segments, cmap='viridis')
plt.axis('off')
plt.show()
