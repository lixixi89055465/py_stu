import torch

a = torch.tensor([1., 2., 3.],requires_grad=True)
b = a.clone()

print(a.data_ptr()) # 1597834661312
print(b.data_ptr()) # 1597834659712 # 内存地址不同

print(a) # tensor([1., 2., 3.], requires_grad=True)
print(b) # tensor([1., 2., 3.], grad_fn=<CopyBackwards>) # 复制成功
print('-'*30)

c = a * 2
d = b * 3

c.sum().backward()
print(a.grad) # tensor([2., 2., 2.])

d.sum().backward()
print('0'*100)
print(a.grad) # tensor([5., 5., 5.]) # 源tensor梯度累加了
print(b.grad) # None # 复制得到的节点依然不是叶子节点


# import torch
#
# a = torch.tensor([1., 2., 3.],requires_grad=True)
# b = torch.empty_like(a).copy_(a)
#
# print(a.data_ptr()) # 1597834661312
# print(b.data_ptr()) # 1597834659712 # 内存地址不同
#
# print(a) # tensor([1., 2., 3.], requires_grad=True)
# print(b) # tensor([1., 2., 3.], grad_fn=<CopyBackwards>) # 复制成功
# print('-'*30)
#
# c = a * 2
# d = b * 3
#
# c.sum().backward()
# print(a.grad) # tensor([2., 2., 2.])
#
# d.sum().backward()
# print(a.grad) # tensor([5., 5., 5.]) # 源tensor梯度累加了
# print(b.grad) # None # 复制得到的节点依然不是叶子节点