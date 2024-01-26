# -*- coding: utf-8 -*-
# @Time : 2024/1/24 22:18
# @Author : nanji
# @Site : 
# @File : testtorch_pow.py
# @Software: PyCharm 
# @Comment : 

# X = torch.arange(max_len, dtype=torch.float32) \
# 		.reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
import torch
num_hiddens=32
arange01=torch.arange(0,32,2,dtype=torch.float32)
arange01/=num_hiddens
pow01=torch.pow(10000,arange01)
max_len=1000
arange02=torch.arange(max_len,dtype=torch.float32)
print(arange02.shape)
result01=arange02.reshape(-1,1)/pow01
print('0'*100)
print(result01)

