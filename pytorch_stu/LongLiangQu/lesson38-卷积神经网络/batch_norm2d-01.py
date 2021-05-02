import torch

# image Normalization
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.504],
#                                  std=[0.229, 0.224, 0.225])

x = torch.rand(1, 16, 7, 7)
layer = torch.nn.BatchNorm2d(16)
out = layer(x)
print(out.shape)
print(layer.weight)
print(layer.weight.shape)
print(layer.bias.shape)
print('1'*20)
print(vars(layer))
print(layer.eval())
out=layer(x)
print(out.shape)
