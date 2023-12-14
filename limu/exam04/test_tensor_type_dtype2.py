# -*- coding: utf-8 -*-
import torch as th

data = th.tensor([[1,2], [3,4]])
print(data.device)  # cpu
print(data.storage()) #1.0 2.0 3.0 4.0 [torch.FloatStorage of size 4]
print(data.data_ptr()) # 2405763261376
print(data.size()) # torch.Size([2, 2])
print(data.shape) # torch.Size([2, 2])
print(data.dtype) # torch.float32
print(data.is_contiguous()) # True
print('0'*100)
data_t = data.t() # Transpose
print(data.data_ptr() == data_t.data_ptr()) # True
print('1'*100)
print(data_t.is_contiguous()) # False
print('2'*100)
data2 = th.ones(4, 2, dtype=th.float32) # 使用create-fucntion创建时指定dtype
data_float32 = th.tensor([[1,2], [3,4]]).double() #强行cast到指定类型
data_float32_2 = th.tensor([[1,2], [3,4]]).to(th.double) # 通过to (更加general的方法)
print('3'*100)
data_t_contig = data_t.contiguous()
print(data_t_contig.is_contiguous())
print(data_t_contig.storage())
print('4'*100)
print(data.data_ptr == data_t_contig.data_ptr)
