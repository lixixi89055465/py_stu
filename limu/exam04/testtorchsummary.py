# -*- coding: utf-8 -*-
# @Time    : 2023/12/10 下午5:57
# @Author  : nanji
# @Site    : 
# @File    : testtorchsummary.py
# @Software: PyCharm 
# @Comment :
from torchsummary import summary
from torchvision.models import vgg16
import torch

device=torch.device('cuda:0')
myNet = vgg16().to(device)
print('0' * 100)
summary(myNet,input_size=(3,64,64))
