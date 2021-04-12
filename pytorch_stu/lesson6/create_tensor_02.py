import torch

print(torch.arange(0, 10))

print(torch.arange(0, 10, 2))

print(torch.linspace(0, 10, steps=4))
print(torch.linspace(0, 10, steps=10))
print(torch.linspace(0, 10, steps=11))
print(torch.logspace(0, -1, steps=10))
print(torch.logspace(0, 1, steps=8))
print(torch.eye(3 ))
# 随机打散
a=torch.rand(2,3)
print(a)
b=torch.rand(2,2)
print(b)
idx=torch.randperm(2)


print(idx)
print(a[idx])

print(b[idx])

print(a, b)

