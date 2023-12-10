import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np

train_dataset = dsets.MNIST(root='./data',  # 文件存放路径
							train=True,  # 提取训练集
							transform=transforms.ToTensor(),  # 将图像转化为Tensor
							download=True)

sample_size = len(train_dataset)
sampler1 = torch.utils.data.sampler.SubsetRandomSampler(
	np.random.choice(range(len(train_dataset)), sample_size))

print(sampler1)
print('0'*100)
for i ,v in enumerate(sampler1):
	print(i, v)
	break
