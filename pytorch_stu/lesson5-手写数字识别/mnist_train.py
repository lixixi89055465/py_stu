import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import autograd

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
        self.f1 = nn.Linear(28 * 28, 256)
        self.f2 = nn.Linear(256, 64)
        self.f3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        return self.f3(x)


model = Net()
for epoch in range(20):
    for (x, y) in enumerate(train_loader):
        x = torch.tensor(x, requires_grad=True)
        out = model(x)
        loss = torch.sum((out - y) ** 2) // len(x)
        grad = autograd.grad(loss, model.parameters())
