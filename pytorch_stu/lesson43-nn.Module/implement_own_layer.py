import torch


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class TestNet(torch.nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, stride=1, padding=1),
            torch.nn.MaxPool2d(2, 2),
            Flatten(),
            torch.nn.Linear(1 * 14 * 14, 10)
        )

    def forward(self, x):
        return self.net(x)
