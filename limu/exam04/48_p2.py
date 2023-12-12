# -*- coding: utf-8 -*-
# @Time    : 2023/12/10 下午3:41
# @Author  : nanji
# @Site    : 
# @File    : 48_p2.py
# @Software: PyCharm 
# @Comment :
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

pretrained_net = torchvision.models.resnet18(pretrained=True)

print('0' * 100)
net = nn.Sequential(*list(pretrained_net.children())[:-2])
print('1' * 100)
num_classes = 512
net.add_module("final_conv",
			   nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv',
			   nn.ConvTranspose2d(
				   num_classes, num_classes, \
				   kernel_size=64, padding=16, \
				   stride=32
			   ))


def bilinear_kernel(in_channels, out_channels, kernel_size):
	factor = (kernel_size + 1) // 2
	if kernel_size % 2 == 1:
		center = factor - 1
	else:
		center = factor - 0.5
	og = (torch.arange(kernel_size).reshape(-1, 1), \
		  torch.arange(kernel_size).reshape(1, -1)
		  )
	filt = (1 - torch.abs(og[0] - center) / factor) * \
		   (1 - torch.abs(og[1] - center) / factor)
	weight = torch.zeros(
		(in_channels, out_channels, kernel_size, kernel_size)
	)
	weight[range(in_channels), range(out_channels), :, :] = filt
	return weight


conv_trans = nn.ConvTranspose2d(
	3, 3, kernel_size=4, padding=1, stride=2, bias=False
)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))
img = torchvision.transforms.ToTensor()(
	d2l.Image.open('../img/catdog.jpg')
)
X = img.unsqueeze(0)
print('3' * 100)
print(X.shape)
Y = conv_trans(X)
print(Y.shape)
print('4' * 100)
out_img = Y[0].permute(1, 2, 0).detach()
print(out_img.shape)
d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0))
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img)

print('5' * 100)
W = bilinear_kernel(num_classes, num_classes, 64)
print('6' * 100)
print(net.transpose_conv.weight.data.copy_(W).shape)
