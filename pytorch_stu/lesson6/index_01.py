import torch

a = torch.rand(4, 3, 28, 28)
print(a[0].shape)
print(a[0, 0].shape)
print(a[0, 0, 24, 24])
print(a.shape)
print(a[:2].shape)
print(a[:2, :1, :, :].shape)
print(a[2:, :1, :, :].shape)
print(a[2:, -1:, :, :].shape)
print(a[:, :, 0:28:2, 0:28:3].shape)
print(a[:, :, ::2, ::2].shape)
print(a.shape)
print('-'*20)
a=torch.linspace(1, 12, steps=12).view(3,4)
print(a)

b=torch.index_select(a,0,torch.tensor([0,2]))
print(b)
print(a.index_select(0, torch.tensor([0, 2])))

c=torch.index_select(a,1,torch.tensor([1,3]))
print(c)
print('-'*30)
print(a.shape)
print(a[...].shape)
print(a[0, ...].shape)
print(a[:, 1, ...].shape)

print(a[..., :2])
# masked_select()
print('-'*20)
x=torch.randn(3,4)
print(x)
mask=x.ge(0.5)
print(mask)
print(torch.masked_select(x, mask).shape)
src=torch.tensor([[4,3,5],[6,7,8]])
# print(torch.take(src, torch.tensor([0, 2])))
print(torch.take(src, torch.tensor([0, 4, 5])))

