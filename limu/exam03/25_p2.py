# -*- coding: utf-8 -*-
# @Time    : 2023/11/29 下午10:01
# @Author  : nanji
# @Site    : 
# @File    : 25_p2.py
# @Software: PyCharm 
# @Comment :
import torch
from torch import nn
from d2l import torch as d2l


def vgg_block(conv_nums, in_channels, out_channels):
    block = []
    for i in range(conv_nums):
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        block.append(nn.ReLU())
        in_channels = out_channels
    block.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*block)


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


def vgg(conv_arch):
    in_channels = 1
    conv_blks = []
    for (num_convs, out_channel) in conv_arch:
        conv_blks.append(
            vgg_block(num_convs, in_channels, out_channel)
        )
        in_channels = out_channel
    return nn.Sequential(
        *conv_blks, nn.Flatten(), nn.Linear(out_channel * 7 * 7, 4096), \
        nn.ReLU(), nn.Dropout(p=0.5), \
        nn.Linear(4096, 4096), \
        nn.ReLU(), nn.Dropout(p=0.5), \
        nn.Linear(4096, 10)
    )


x = torch.randn(size=(1, 1, 224, 224))
net = vgg(conv_arch)
for blk in net:
    x = blk(x)
    print(blk.__class__.__name__, 'out shape:\t', x.shape)

ratio = 1
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]

net = vgg(small_conv_arch)

batch_size, resize = 128, 224
epoch_num, lr = 10, 0.05
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=resize)
d2l.train_ch6(net, train_iter, test_iter, epoch_num, lr, device=d2l.try_gpu())
