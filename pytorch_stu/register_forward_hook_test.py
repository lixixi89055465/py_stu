'''

'''
import torch
import torch.nn as nn
net1=nn.Linear(4,2)
net2=nn.Linear(2,1)

def hook_func(model,input,output):
    print(model)
    print('input:',input)
    print('output:',output)

x=torch.tensor([[1.,2.,3.,4.]],requires_grad=True)
handles1=net1.register_forward_hook(hook_func)
print(handles1)

y=net1(x)
print('y：',y)
