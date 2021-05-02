import torch
from torch import nn


class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 20),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 784),
            torch.nn.Sigmoid()
        )

        self.criteon = torch.nn.MSELoss()

    def forward(self, x):
        batchsz = x.size(0)
        x = x.view(batchsz, 784)
        h_ = self.encoder(x)
        mu, sigma = h_.chunk(2, dim=1)
        h = mu + sigma * torch.randn_like(sigma)
        x_hat = self.decoder(h)
        x_hat = x_hat.view(batchsz, 1, 28, 28)
        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) +
            torch.pow(sigma, 2) -
            torch.log(1e-8 + torch.pow(sigma, 2)) - 1
        ) / (batchsz * 28 * 28)
        return x_hat, kld
