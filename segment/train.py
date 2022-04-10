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


from utils import visualize_mean_color_image
visualize_mean_color_image(img, segments)
print('2'*100)

from utils import load_dataset, compute_segmentation
from segmentation import evaluate_segmentation

# 载入该小型数据集
imgs, gt_masks = load_dataset('./data')

# 设置图像分割的参数
num_segments = 3
clustering_fn = kmeans_fast
feature_fn = color_features
scale = 0.5

mean_accuracy = 0.0

segmentations = []

for i, (img, gt_mask) in enumerate(zip(imgs, gt_masks)):
    # Compute a segmentation for this image
    segments = compute_segmentation(img, num_segments,
                                    clustering_fn=clustering_fn,
                                    feature_fn=feature_fn,
                                    scale=scale)

    segmentations.append(segments)

    # 评估图像分割结果
    accuracy = evaluate_segmentation(gt_mask, segments)

    print('Accuracy for image %d: %0.4f' % (i, accuracy))
    mean_accuracy += accuracy

mean_accuracy = mean_accuracy / len(imgs)
print('Mean accuracy: %0.4f' % mean_accuracy)

