import torch

a = torch.randn(2, 3)
print(a.type())
print(type(a))
print(isinstance(a, torch.FloatTensor))
print(isinstance(a, torch.cuda.DoubleTensor))

data = a.cuda()
print(data)
print(isinstance(data, torch.cuda.FloatTensor))

print('-' * 30)
a=torch.tensor(2.2)
print(torch.tensor(1.))
print(torch.tensor(1.3))
print(torch.tensor(2.2))
print(a.shape)
print(len(a.shape))

print(len(a.shape))
print(a.size())
