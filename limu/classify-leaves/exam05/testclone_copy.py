import torch

a = torch.tensor([1.,2.,3.],requires_grad=True)
b = a.clone()

print(a.data_ptr()) # 3004830353216
print(b.data_ptr()) # 3004830353344 内存地址不同

print(a) # tensor([1., 2., 3.], requires_grad=True)
print(b) # tensor([1., 2., 3.], grad_fn=<CloneBackward>)  复制成功
print('-'*30)

c = a * 2
d = b * 3

c.sum().backward()
print(a.grad) # tensor([2., 2., 2.])

d.sum().backward()
print('2'*100)
print(a.grad) # tensor([5., 5., 5.]) # 源tensor的梯度叠加了新tensor的梯度
print(b.grad) # None # 此时复制出来的节点已经不属于叶子节点，因此不能直接得到其梯度