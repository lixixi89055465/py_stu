# -*- coding: utf-8 -*-
# @Time    : 2023/12/10 下午5:57
# @Author  : nanji
# @Site    : 
# @File    : testtorchsummary.py
# @Software: PyCharm 
# @Comment :https://blog.csdn.net/Wenyuanbo/article/details/118514709
from torchsummary import summary
from torchvision.models import vgg16
import torch

device=torch.device('cpu')
myNet = vgg16().to(device)
print('0' * 100)
summary(myNet,input_size=(3,64,64),device='cpu')
