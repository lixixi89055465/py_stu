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
from d2l import torch as d2l
import matplotlib.pyplot as plt
import time

# d2l.use_svg_display()
# 通过ToTensor示例将图像数据从PIL 类型变成32位浮点数格式
# 并除以255使得所有像素的数值均在0到1至今
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root='./data', \
                                                train=True, \
                                                transform=trans, \
                                                download=True
                                                )
minist_test = torchvision.datasets.FashionMNIST(root='./data', \
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
    axes = plt.figure(figsize=figsize)
    for i, img in enumerate(imgs):
        # plt.subplot(291 + i)
        ax = axes.add_subplot(2, 9, i + 1)
        if torch.is_tensor(img):
            # 图片张良
            ax.imshow(img.numpy())
        else:
            # PIL 图片
            ax.imshow(img)


X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
plt.savefig('1.png')
plt.show()
batch_size = 256


def get_dataloader_worker():
    '''使用4各进程来读取的数据。'''
    return 0


train_iter = data.DataLoader(mnist_train, \
                             batch_size=batch_size, \
                             shuffle=True, \
                             num_workers=get_dataloader_worker())
# timer = d2l.Timer()
start = time.time()
for X, y in train_iter:
    continue
end = time.time()
print(f'{(end - start):.2f} sec')


def load_data_fashion_mnist(batch_size, resize=None):
    '''
    下载Fashion-MNIST数据集，然后将其加载到内存中。
    :param batch_size:
    :param resize:
    :return:
    '''
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='./data', \
                                                    train=True, transform=trans, \
                                                    download=True,
                                                    )
    mnist_test = torchvision.datasets.FashionMNIST(root='./data', \
                                                   train=False, transform=trans, \
                                                   download=True)
    return (data.DataLoader(mnist_train, batch_size=batch_size, \
                            shuffle=True, num_workers=get_dataloader_worker()), \
            data.DataLoader(mnist_test, batch_size=batch_size, \
                            shuffle=False, num_workers=get_dataloader_worker()))


from IPython import display

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
print('0' * 100)
print(b)


def softmax(x):
    x_exp = torch.exp(x)
    partition = x_exp.sum(dim=1, keepdim=True)
    return x_exp / partition  # 这里应用了广播机制


x = torch.normal(0, 1, size=(2, 5))
x_prob = softmax(x)
print(x_prob)
print('1' * 100)
print(x_prob)
print(x_prob.sum())


def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print('2' * 100)
print(y_hat[[0, 1], y])


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


print('3' * 100)
print(cross_entropy(y_hat, y))


def accuracy(y_hat, y):
    '''计算预测正确的数量。'''
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


print(accuracy(y_hat, y) / len(y))


def evaluate_accuracy(net, data_iter):
    '''
    计算在指定数据集上模型的精度。
    :param net:
    :param data_iter:
    :return:
    '''
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数，预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


class Accumulator:
    '''在n个变量上累加 '''

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


evaluate_accuracy(net, test_iter)


def loss(y_hat, y):
    return -torch.log(y_hat[range(len(y)), torch.tensor(y)])


def train_epoch_ch3(net, train_iter, test_iter, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for x, y in train_iter:
        y_hat = net(x)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(
                (float(l) * len(y),
                 accuracy(y_hat, y),
                 y.size().numel()),
            )
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), \
                       y.size().numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = d2l.Animator(x_label='epoch', xlim=[1, num_epochs], \
                            ylim=[0.3, 0.9], \
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc))
    train_loss, train_acc = train_metrics
    return train_loss, train_acc


lr = 0.1


def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
num_epochs=10

train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)