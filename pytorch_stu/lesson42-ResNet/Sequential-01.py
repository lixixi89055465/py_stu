import torch

net = torch.nn.Sequential(
    torch.nn.Conv2d(1, 32, 5, 1, 1),
    torch.nn.MaxPool2d(2, 2),
    torch.nn.ReLU(True),
    torch.nn.BatchNorm2d(32),

    torch.nn.Conv2d(32, 64, 3, 1, 1),
    torch.nn.ReLU(True),
    torch.nn.BatchNorm2d(64),

    torch.nn.Conv2d(64, 64, 3, 1, 1),
    torch.nn.MaxPool2d(2, 2),
    torch.nn.ReLU(True),
    torch.nn.BatchNorm2d(64),

    torch.nn.Conv2d(64, 128, 3, 1, 1),
    torch.nn.ReLU(True),
    torch.nn.BatchNorm2d(128)
)

net = torch.nn.Sequential(
    torch.nn.Linear(4, 2),
    torch.nn.Linear(2, 2)
)
print(list(net.parameters())[0].shape)
print(list(net.parameters())[1].shape)

print(dict(net.named_parameters()).items())

optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
