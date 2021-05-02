import torch

a = torch.rand(4, 10) * 2 - 1;
print(a)
# print(a>0)
# print(torch.gt(a, 0))

# print(a!=0)

a = torch.ones(2, 3)
print(a.shape)
b = torch.randn(2, 3)
print(b)
print(torch.eq(a, b))
print(a > 0)
print(torch.gt(a, 0))

print(a != 0)
a = torch.ones(2, 3)
print(a)
a = torch.ones(2, 3)
print(a)
b = torch.randn(2, 3)
print(b)
print(torch.eq(a, a))
print('-' * 20)
print(torch.equal(a, a))
