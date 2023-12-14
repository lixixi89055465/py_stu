import torch

x = torch.tensor([1, 2])
print(x.dtype)
print(x.type())
x = torch.tensor([1, 2, 4])
print('0' * 100)
print(x.dtype)
y = torch.tensor([1., 2., 3.])
print(y.dtype)

print('1' * 100)
x = torch.tensor([1, 2, 3], dtype=torch.int8)
print(x.dtype)
y = torch.CharTensor([1, 2, 3])
print(y.dtype)
print(y)
print('2' * 100)
x = torch.Tensor([1, 2])
print(x.type())
y = torch.tensor([1, 2], dtype=torch.long)
print(y.type())

print('3' * 100)
x = torch.tensor([1, 2, 3])
x = x.short()
print(x.dtype)
print('4' * 100)
y = torch.tensor([1, 2, 3])
y = y.type(torch.int64)
print(y.dtype)
# 2. Tensor存储结构
x=torch.tensor([1,2,3,4,5,6])
y=x.reshape(2,3)
print('5'*100)
print(x.storage(), y.storage())
print('6'*100)
print(x.storage().data_ptr())
print('7'*100)
print(y.storage().data_ptr())
print('8'*100)
print(x.stride())
print('9'*100)
print(y.stride())
