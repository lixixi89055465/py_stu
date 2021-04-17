import torch

# image Normalization
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.504],
#                                  std=[0.229, 0.224, 0.225])

x = torch.rand(100, 16, 784)
layer = torch.nn.BatchNorm1d(16)
out = layer(x)
print(out.shape)
print(layer.running_mean)
print(layer.running_var)
