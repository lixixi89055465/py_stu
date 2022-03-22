'''

'''
import torch
data1=[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 0], [10, 11]]]
t1=torch.Tensor(data1).long()
print(t1)
data2= [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [10, 11]]]
t2=torch.Tensor(data2).long()
print(t2)
print(t1.size())
print(t2.size())
print('-------view  as ()-------')
t2=t2.view_as(t1)
print(t2)
print(t2.size())
