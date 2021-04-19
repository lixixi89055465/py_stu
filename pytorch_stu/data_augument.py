import torch
import torchvision

cifar_data = torchvision.datasets.CIFAR10('cifar', True, transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
]), download=True)

cifar_train = torch.utils.data.DataLoader(
    cifar_data,
    batch_size=128,
    shuffle=True
)

cifar_test = torchvision.datasets.CIFAR10('cifar', False, transform=torchvision.transforms.Compose([

    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
]), download=True)
