import torch
# from torch import nn
# from torch.nn import functional as F
# from torch import optim
# from torch import autograd

import torchvision
from matplotlib import pyplot as plt
from utils import plot_image, plot_curve, one_hot

# print(torch.version)
print(torch.__version__)
# step 1 load dataset
batch_size = 128
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data/', train=True,
                               download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   # torchvision.transforms.Normalize(
                                   #     (0.1307,), (0.3081,))
                               ])), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data/', train=False,
                               download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])), batch_size=batch_size, shuffle=False)

print("test")

(x, y) = next(iter(train_loader))
print(x.shape, y.shape, x.min(), x.max())
plot_image(x, y, 'image_sample')


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.f1 = torch.nn.Linear(28 * 28, 256)
        self.f2 = torch.nn.Linear(256, 64)
        self.f3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.f1(x))
        x = torch.nn.functional.relu(self.f2(x))
        return self.f3(x)


model = Net()
train_loss = []
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
for epoch in range(20):
    for step, (x, y) in enumerate(train_loader):
        x = x.view(x.size(0), 28 * 28)
        out = model(x)
        y_onehot = one_hot(y)
        loss = torch.nn.functional.mse_loss(out, y_onehot)
        # 梯度清零
        optimizer.zero_grad()
        loss.backward()
        # 更新梯度
        optimizer.step()
        train_loss.append(loss)

# we get optimal [w1,b1,w2,b2,w3,b3]
total_correct = 0.
plot_curve(train_loss)
for x, y in test_loader:
    x = x.view(x.size(0), 28 * 28)
    out = model(x)
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct
total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc', acc)
x, y = next(iter(test_loader))
out = model(x.view(x.size(0), 28 * 28))
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')
