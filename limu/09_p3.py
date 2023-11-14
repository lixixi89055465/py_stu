# -*- coding: utf-8 -*-
# @Time    : 2023/11/14 22:18
# @Author  : nanji
# @Site    : 
# @File    : 09_p3.py
# @Software: PyCharm 
# @Comment :
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
# from d2l import torch as d2l
import matplotlib.pyplot as plt
import time
# d2l.use_svg_display()
# 通过ToTensor示例将图像数据从PIL 类型变成32位浮点数格式
# 并除以255使得所有像素的数值均在0到1至今
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root='./data/', \
                                                train=True, \
                                                transform=trans, \
                                                download=True
                                                )
minist_test = torchvision.datasets.FashionMNIST(root='./data/', \
                                                train=False, \
                                                transform=trans, \
                                                download=True)

print(len(mnist_train))
print(len(minist_test))

print(mnist_train[0][0].shape)
print('0' * 100)
print(mnist_train.classes)


def get_fashion_mnist_labels(labels):
    '''
    返回fashion-MNIST数据集的文本标签
    :param labels:
    :return:
    '''
    text_labels = ['T-shirt/top', 'Trouser', 'Pullover', \
                   'Dress', 'Coat', 'Sandal', 'Shirt', \
                   'Sneaker', 'Bag', 'Ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=2):
    '''plot a list of images '''
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.figure( figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        plt.subplot(291+i)
        if torch.is_tensor(img):
            # 图片张良
            ax.imshow(img.numpy())
        else:
            # PIL 图片
            ax.imshow(img)


X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
plt.show()
batch_size = 256


def get_dataloader_worker():
    '''使用4各进程来读取的数据。'''
    return 4


train_iter = data.DataLoader(mnist_train, batch_size=batch_size, \
                             shuffle=True, num_workers=get_dataloader_worker())
# timer = d2l.Timer()
start=time.time()
for X, y in train_iter:
    continue
end=time.time()
print(f'{(end-start):.2f} sec')
