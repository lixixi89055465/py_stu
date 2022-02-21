'''

'''
import torch
# 矩阵 转置
a=torch.arange(6).reshape(2,3)
b=torch.einsum('ij->ji',a)
print(a.shape)
print(b.shape)
# 求和
a=torch.arange(6).reshape(2,3)
b=torch.einsum('ij->',a)
print(a.shape)
print(b.shape)
# 列求和
a=torch.arange(6).reshape(2,3)
b=torch.einsum('ij->j',a)
print(a.shape)
print(b.shape)
# 行求和
a=torch.arange(6).reshape(2,3)
b=torch.einsum('ij->i',a)
print(b.shape)
print(b)
# 矩阵相乘
a=torch.arange(6).reshape(2,3)
b=torch.arange(3)
c=torch.einsum('ik,k->i',a,b)
print(c.shape)
print(c)
a=torch.arange(6).reshape(2,3)
b=torch.arange(15).reshape(3,5)
c=torch.einsum('ik,kj->ij',a,b)
print(c.shape)

a=torch.arange(3)
b=torch.arange(3,6)
c=torch.einsum('i,i->i',[a,b])
print(c.shape)
print(c)
# 矩阵
a=torch.arange(6).reshape(2,3)
b=torch.arange(6,12).reshape(2,3)
c=torch.einsum('ij,ij->',[a,b])
print(c)
# 哈达玛积
a=torch.arange(6).reshape(2,3)
b=torch.arange(6,12).reshape(2,3)
c=torch.einsum('ij,ij->ij',[a,b])
print(c)
# 外积
a=torch.arange(3)
b=torch.arange(3,7)
c=torch.einsum('i,j->ij',[a,b])
print(c)
# batch 矩阵相乘
a=torch.randn(3,2,5)
b=torch.randn(3,5,3)
c=torch.einsum('ijk,ikl->ijl',[a,b])
print(c)
# 张量缩约
a=torch.randn(2,3,5,7)
b=torch.randn(11,13,3,17,5)
c= torch.einsum('pqrs,tuqvr->pstuv',[a,b])
print(c.shape)

# 双线性变换
a=torch.randn(2,3)
b=torch.randn(5,3,7)
c=torch.randn(2,7)
d= torch.einsum('ik,jkl,il->ij',[a,b,c])
print(d.shape)
# 案例
