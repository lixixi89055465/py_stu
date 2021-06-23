import torch


class Encoder(torch.nn.Module):
    def __init__(self, im_chan=1, output_chan=32, hidden_dim=16):
        super(Encoder, self).__init__()
        self.z_dim = output_chan
        self.disc = torch.nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, output_chan * 2, final_layer=True),
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        if not final_layer:
            return torch.nn.Sequential(
                torch.nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                torch.nn.BatchNorm2d(output_channels),
                torch.nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            return torch.nn.Sequential(
                torch.nn.Conv2d(input_channels, output_channels, kernel_size, stride)
            )

    def forward(self, image):
        disc_pred = self.disc(image)
        encoding = disc_pred.view(len(disc_pred), -1)
        return encoding[:, :self.z_dim], encoding[:, self.z_dim:].exp()


class Decoder(torch.nn.Module):
    def __init__(self, z_dim=32, im_chan=1, hidden_dim=64):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.gen = torch.nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                torch.nn.BatchNorm2d(output_channels),
                torch.nn.ReLU(inplace=True),
            )
        else:
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                torch.nn.Sigmoid()
            )

    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)


class VAE(torch.nn.Module):
    def __init__(self, z_dim=32, im_chan=1, hidden_dim=64):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encode = Encoder(im_chan, z_dim)
        self.decode = Decoder(z_dim, im_chan)

    def forward(self, images):
        q_mean, q_stddev = self.encode(images)
        q_dist = torch.distributions.normal.Normal(q_mean, q_stddev)
        z_sample = q_dist.rsample()
        decoding = self.decode(z_sample)
        return decoding, q_dist


reconstruction_loss = torch.nn.BCELoss(reduction='sum')


def kl_divergence_loss(q_dist):
    return torch.distributions.kl.kl_divergence(
        q_dist, torch.distributions.normal.Normal(torch.zeros_like(q_dist.mean), torch.ones_like(q_dist.stddev))
    ).sum(-1)


# from torch.utils.data.dataloader import DataLoader
# from torchvision import datasets, transforms
import torchvision

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
mnist_dataset = torchvision.datasets.MNIST('../data/', train=True, download=True, transform=transforms)
train_dataloader = torch.utils.data.DataLoader(mnist_dataset, shuffle=True, batch_size=1024)

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (16, 8)

from torchvision.utils import make_grid
from tqdm import tqdm
import time


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.axis('off')
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())


device = 'cuda'
vae = VAE().to(device)
vae_opt = torch.optim.Adam(vae.parameters(), lr=0.002)
for epoch in range(10):
    print(f"Epoch{epoch}")
    time.sleep(0.5)
    for images, _ in tqdm(train_dataloader):
        images = images.to(device)
        vae_opt.zero_grad()
        recon_images, encoding = vae(images)
        loss = reconstruction_loss(recon_images, images) + kl_divergence_loss(encoding).sum()
        loss.backward()
        vae_opt.step()
    plt.subplot(1, 2, 1)
    show_tensor_images(images)
    plt.title('True')
    plt.subplot(1, 2, 2)
    show_tensor_images(recon_images)
    plt.title('Reconstructed')
    plt.show()
