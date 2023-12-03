# -*- coding: utf-8 -*-
# @Time    : 2023/12/3 上午9:53
# @Author  : nanji
# @Site    : 
# @File    : 34_p2.py
# @Software: PyCharm 
# @Comment :
import torch
from torch import nn
from d2l import torch as d2l


def resnet18(num_classes, in_channels=1):
    def resnet_block(in_channels, out_channels, num_residuals, \
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(
                    d2l.Residual(in_channels, out_channels, use_1x1conv=True, strides=2)
                )
            else:
                blk.append(
                    d2l.Residual(out_channels, out_channels)
                )
        return nn.Sequential(*blk)

    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        resnet_block(64, 64, 2, first_block=True),
        resnet_block(64, 128, 2),
        resnet_block(128, 256, 2),
        resnet_block(256, 512, 2),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, num_classes)
    )
    return net


def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]

    def init_weight(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weight)
    net = nn.DataParallel(net, device_ids=devices, )
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter, devices[0])))
    print(f'test acc:{animator.Y[0][-1]:.2f}, {timer.sum():.1f} examples /sec '
          f' on {str(devices)}')


net = resnet18(10, 1)
X = torch.rand(1, 1, 224, 224)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output size:', X.shape)
print('0' * 100)
batch_size, lr = 128, 0.1
train(resnet18(10, 1), num_gpus=1, batch_size=batch_size, lr=lr)
