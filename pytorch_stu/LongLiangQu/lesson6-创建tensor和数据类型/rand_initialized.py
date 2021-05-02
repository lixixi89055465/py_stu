import torch

print(torch.rand(3, 3))
a = torch.rand(3, 3)
print(a)

print(torch.rand_like(a))
# [min,max)
print(torch.randint(1, 10, (3, 3)))
# normal distribution
print('-' * 20)
print(torch.randn(3, 3))
# print(torch.normal(mean=torch.full([10], 0), std=torch.arange(1, 0, -0.1)))
print(torch.full([10], 0))

print(torch.normal(mean=torch.full([10], 0.), std=torch.arange(1, 0, -0.1)))
print('-'*20)
print(torch.full([2, 3], 7))
print(torch.full([], 7))
print(torch.full([1], 7))

