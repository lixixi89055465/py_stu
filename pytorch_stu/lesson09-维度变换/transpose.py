import torch

a = torch.rand(4, 3, 32, 32)
print(a.shape)
a1 = a.transpose(2, 1)
print(a1.shape)
# a1=a.transpose(1,3).view(4,3*32*32).view(4,3,32,32)
# print(a1.shape)
a1 = a.transpose(1, 3)
print(a1.shape)
print(a.shape)
a1 = a.transpose(1, 3).contiguous().view(4, 3 * 32 * 32).view(4, 3, 32, 32)
a2 = a.transpose(1, 3).contiguous().view(4, 3 * 32 * 32).view(4, 32, 32, 3).transpose(1, 3)
print(a1.shape)
print(a2.shape)

print(torch.all(torch.eq(a, a1)))
print(torch.all(torch.eq(a, a2)))
