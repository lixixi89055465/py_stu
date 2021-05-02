import torch

x = torch.arange(12, dtype=torch.int32).reshape((3, 4))
y = torch.ones(12, dtype=torch.int32).reshape((3, 4))
print(x)
print(x[-1])
print(x[1:3])

print(x[-1], x[1:3])
print('*' * 20)

print(x[1, 2])
x[1, 2] = 9
print(x)
# python 中是左开右闭
x[0:2, :] = 12
print(x)
print(x[-1])
print(x[1:3])

before = id(y)
print(before)
y = y + x
print(id(y))
print(id(y) == before)

# 不申请内存，进行id共享
z = torch.zeros_like(y)
print('id(z):', id(z))
z[:] = x + y
print('id(z):', id(z))
# 需要重新申请内存
z = x + y
print('id(z):', id(z))
before = id(x)
x += y
print(id(x) == before)
x[:] = x + y
print(id(x) == before)
A = x.numpy()
print(A)
B = torch.tensor(A)
print(B)
print(type(A), type(B))

# 将大小为1的张量转化为python标量
a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))
