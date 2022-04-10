# segmentation.py
# python 3.6
import numpy as np
import random
from scipy.spatial.distance import squareform, pdist
from skimage.util import img_as_float

### Clustering Methods
def kmeans(features, k, num_iters=100):

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]        # 1. 随机中心点
    assignments = np.zeros(N)

    for n in range(num_iters):
        ### YOUR CODE HERE
        # 2. 分类
        for i in range(N):
            dist = np.linalg.norm(features[i] - centers, axis=1)    # 每个点和中心点的距离
            assignments[i] = np.argmin(dist)        # 第i个点属于最近的中心点

        pre_centers = centers.copy()
        # 3. 重新计算中心点
        for j in range(k):
            centers[j] = np.mean(features[assignments == j], axis=0)

        # 4. 验证中心点是否改变
        if np.array_equal(pre_centers, centers):
            break
        ### END YOUR CODE

    return assignments

def kmeans_fast(features, k, num_iters=100):

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    for n in range(num_iters):
        ### YOUR CODE HERE
        # 计算距离
        features_tmp = np.tile(features, (k, 1))        # (k*N, ...)
        centers_tmp = np.repeat(centers, N, axis=0)     # (N * k, ...)
        dist = np.sum((features_tmp - centers_tmp)**2, axis=1).reshape((k, N))      # 每列 即k个中心点
        assignments = np.argmin(dist, axis=0)   # 最近

        # 计算新的中心点
        pre_centers = centers
        # 3. 重新计算中心点
        for j in range(k):
            centers[j] = np.mean(features[assignments == j], axis=0)

        # 4. 验证中心点是否改变
        if np.array_equal(pre_centers, centers):
            break
        ### END YOUR CODE

    return assignments



def hierarchical_clustering(features, k):

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N)
    centers = np.copy(features)
    n_clusters = N

    while n_clusters > k:
        ### YOUR CODE HERE
        dist = pdist(centers)       # 计算相互之间的距离
        matrixDist = squareform(dist)   # 将向量形式变化为矩阵形式
        matrixDist = np.where(matrixDist != 0.0, matrixDist, 1e10)      # 将0.0的变为1e10,即为了矩阵中相同的点计算的距离去掉

        minValue = np.argmin(matrixDist)        # 最小的值的位置
        min_i = minValue // n_clusters          # 行号
        min_j = minValue - min_i * n_clusters   # 列号

        if min_j < min_i:       # 归并到小号的cluster
            min_i, min_j = min_j, min_i  # 交换一下

        for i in range(N):
            if assignments[i] == min_j:
                assignments[i] = min_i     # 两者合并

        for i in range(N):
            if assignments[i] > min_j:
                assignments[i] -= 1     # 合并了一个cluster,因此n_clusters减少一位

        centers = np.delete(centers, min_j, axis=0)  # 减少一个
        centers[min_i] = np.mean(features[assignments == min_i], axis=0)        # 重新计算中心点

        n_clusters -= 1     # 减去1

        ### END YOUR CODE

    return assignments


### Pixel-Level Features
def color_features(img):
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    ### YOUR CODE HERE
    features = img.reshape(H * W, C)        # color作为特征
    ### END YOUR CODE

    return features

def color_position_features(img):
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    ### YOUR CODE HERE
    # 坐标
    cord = np.dstack(np.mgrid[0:H, 0:W]).reshape((H*W, 2))      # mgrid生成坐标，重新格式为（x,y）的二维
    features[:, 0:C] = color.reshape((H*W, C))      # r,g,b
    features[:, C:C+2] = cord
    a=np.mean(features,axis=0)
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0,  ddof = 0)     # 对特征归一化处理
    ### END YOUR CODE

    return features

def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    features = None
    ### YOUR CODE HERE
    features = color_position_features(img)
    ### END YOUR CODE
    return features


### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    accuracy = None
    ### YOUR CODE HERE
    mask_end = mask_gt - mask
    count = len(mask_end[np.where(mask_end == 0)])
    accuracy = count / (mask_gt.shape[0] * mask_gt.shape[1])
    ### END YOUR CODE

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # 将分割结果与真实值进行对比
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
