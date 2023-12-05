# -*- coding: utf-8 -*-
# @Time    : 2023/12/4 下午9:11
# @Author  : nanji
# @Site    : 
# @File    : 37_p2.py
# @Software: PyCharm 
# @Comment :
import torch
from d2l import torch as d2l
import torchvision
import os
from torch import nn

d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 'fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = d2l.download_extract('hotdog')
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))

test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)

normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225]
)
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(), normalize
])
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(225),
    torchvision.transforms.ToTensor(), normalize
])

pretrained_net = torchvision.models.resnet18(pretrained=True)
print(pretrained_net.fc)
print('0' * 100)
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)


def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=10, \
                      param_group=True):
    train_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), \
                                         transform=train_augs), \
        batch_size=batch_size, \
        shuffle=True
    )
    test_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), \
                                         transform=test_augs), \
        batch_size=batch_size, \
        shuffle=False
    )
    devices = [d2l.try_gpu()]
    loss = nn.CrossEntropyLoss(reduction='none')
    if param_group:
        param_lx = [
            param for name, param in net.named_buffers()
            if name not in ['fc.weight', 'fc.bias']]
        trainer = torch.optim.SGD([
            {'params': param_lx},
            {'params': net.fc.parameters(),
             'lr': learning_rate * 10}
        ], lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(),
                                  lr=learning_rate,
                                  weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, \
                   trainer, num_epochs, devices)


# train_fine_tuning(finetune_net, learning_rate=5e-5)

# 为了进行比较，所有模型参数初始化为随机值
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)
