import torch
from torch import nn

print(torch.device('cpu'), torch.cuda.device('cuda'), torch.cuda.device('cuda:0'))
print('0' * 100)
print(torch.cuda.device_count())


def try_gpu(i=0):
    '''如果存在，则返回gpu(i),否则返回cpu().'''
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())
    ]
    return devices if devices else [torch.device('cpu')]


x = torch.tensor([1, 2, 3])
print('2' * 100)
print(x.device)
print('3' * 100)
# x = torch.ones(2, 3, device=try_gpu())
x = torch.ones(2, 3, device=try_gpu())
print('4' * 100)
print(x)
print('5' * 100)
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
print('6' * 100)
print(net(x))
print('7'*100)
print(net[0].weight.data.device)
print('8'*100)
