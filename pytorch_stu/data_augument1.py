import torch
import torchvision

batch_size = 128
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../images', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.RandomHorizontalFlip(),
                                   torchvision.transforms.RandomVerticalFlip(),  # 随机
                                   torchvision.transforms.RandomRotation(15),  # 随机旋转角度范围
                                   # torchvision.transforms.RandomRotation([90, 180, 270]),
                                   torchvision.transforms.Resize([32, 32]),  # 缩放
                                   torchvision.transforms.RandomCrop([28, 28]),  # 随机减除部分边角,和随机旋转组合
                                   torchvision.transforms.ToTensor()

                               ])), batch_size=batch_size, shuffle=True)
# 白噪声的增强需要自己写代码