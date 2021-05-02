import torch
from torch import nn


class AE(torch.nn.Module):

    def __init__(self):
        super(AE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 20),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(20, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 784),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        batchsz = x.size(0)
        x = x.view(batchsz, 784)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(batchsz, 1, 28, 28)
        return x, None
