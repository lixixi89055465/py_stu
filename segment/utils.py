# utils.py
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage import transform
from skimage import io

from segmentation import *

import os

def visualize_mean_color_image(img, segments):

    img = img_as_float(img)
    k = np.max(segments) + 1
    mean_color_img = np.zeros(img.shape)

    for i in range(k):
        mean_color = np.mean(img[segments == i], axis=0)
        mean_color_img[segments == i] = mean_color

    plt.imshow(mean_color_img)
    plt.axis('off')
    plt.show()

def compute_segmentation(img, k,
        clustering_fn=kmeans_fast,
        feature_fn=color_position_features,
        scale=0):
    """ 计算图像分割结果

    首先，从图像的每个像素中提取一个特征向量。然后将聚类算法应用于所有特征向量的集合。当且仅当两个像素的特征向量被分配到同一簇时，两个像素会被分配到同一聚簇。

    """

    assert scale <= 1 and scale >= 0, \
        'Scale should be in the range between 0 and 1'

    H, W, C = img.shape

    if scale > 0:
        # 缩小图像来获得更快的计算速度
        img = transform.rescale(img, scale)

    features = feature_fn(img)
    assignments = clustering_fn(features, k)
    segments = assignments.reshape((img.shape[:2]))

    if scale > 0:
        # 调整大小分割回图像的原始大小
        segments = transform.resize(segments, (H, W), preserve_range=True)

        # 调整大小会导致像素值不重叠。
        # 像素值四舍五入为最接近的整数
        segments = np.rint(segments).astype(int)

    return segments


def load_dataset(data_dir):
    """
    载入数据集
    'imgs/aaa.jpg' is 'gt/aaa.png'
    """

    imgs = []
    gt_masks = []

    for fname in sorted(os.listdir(os.path.join(data_dir, 'imgs'))):
        if fname.endswith('.jpg'):
            # 读入图像
            img = io.imread(os.path.join(data_dir, 'imgs', fname))
            imgs.append(img)

            # 加载相应的分割mask
            mask_fname = fname[:-4] + '.png'
            gt_mask = io.imread(os.path.join(data_dir, 'gt', mask_fname))
            gt_mask = (gt_mask != 0).astype(int) # 将mask进行二值化
            gt_masks.append(gt_mask)

    return imgs, gt_masks

