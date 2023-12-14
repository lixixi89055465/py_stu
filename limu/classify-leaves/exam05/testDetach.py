import torch

a = torch.tensor([1., 2., 3.], requires_grad=True)
b = a.detach()

print(a.data_ptr())  # 2432102290752
print(b.data_ptr())  # 2432102290752 # 内存位置相同

print(a)  # tensor([1., 2., 3.], requires_grad=True)
print(b)  # tensor([1., 2., 3.]) # 这里为False，就省略了
print('-' * 30)
c = a * 2
d = b * 3
print('1' * 100)
c.sum().backward()
print(a.grad)
print('2' * 100)
print(d.sum().backward())
print(a.grad)
print('3'*100)
print(b.grad)
