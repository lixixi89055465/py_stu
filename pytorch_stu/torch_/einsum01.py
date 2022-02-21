'''

'''
import torch

A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.einsum('ik,kj', A, B)
print(C.shape)
A = torch.randn(3, 4)
B = torch.randn(5, 4)
C = torch.einsum('ik,jk', A, B)
print(C.shape)
C = torch.einsum('ij->', A)
print(C)
C = torch.einsum('ij->j', A)
print(C)
print(C.shape)
# 转置
print(A.shape)
C = torch.einsum('ij->ji', A)
print(C.shape)
# 多维矩阵乘法
print('多维矩阵乘法')
A = torch.randn(3, 4)
B = torch.randn(3, 4, 5)
C = torch.randn(4, 5)
F=torch.einsum('ij,ijk',A,B)
print('1'*100)
print(F.shape)
print(C.shape)
D=torch.einsum('k,jk',F,C)
print(D.shape)
import numpy as np
print('2'*100)
a=np.arange(60.).reshape(3,4,5)
b=np.arange(24.).reshape(4,3,2)
o=np.einsum('ijk,jil->kl',a,b)
print(o.shape)
print(o)
# a[:,:,0]
c=np.sum(a[:,:,0]*b[:,:,1].T)
print(c)



