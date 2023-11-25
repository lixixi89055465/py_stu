import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))


# print('0' * 100)
# print(net(X))
# print('1' * 100)
# # print(net[2].state_dict())
# print(net[0].state_dict())
# print('2' * 100)
# print(type(net[2].bias))
# print('3' * 100)
# print(net[2].bias)
# print('4' * 100)
# print(net[2].bias.data)
#
# print('5' * 100)
# print(net[2].weight.grad == None)
# print('6' * 100)
# print(*[(name, param.shape) for name, param in net[0].named_parameters()])
# print([(name, param.shape) for name, param in net[0].named_parameters()])
# print('7' * 100)
# print(*[(name, param.shape) for name, param in net.named_parameters()])
# print([(name, param.shape) for name, param in net.named_parameters()])
#

# def block1():
#     return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))


# def block2():
#     net = nn.Sequential()
#     for i in range(4):
#         net.add_module(f'block {i}', block1())
#     return net
#
#
# rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
# print('8' * 100)
# print(rgnet(X))
#
# print('9' * 100)
# print([(name, param.shape) for name, param in net.named_parameters()])
# print('10' * 100)
# print(rgnet)


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


net.apply(init_normal)
print('0' * 100)
print(net[0].weight.data[0], net[0].bias.data[0])


def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


def xaview(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


net[0].apply(xaview)
net[2].apply(init_42)
print('2' * 100)
print(net[0].weight.data[0])
print(net[2].weight.data)


def my_init(m):
    if type(m) == nn.Linear:
        print(
            'init',
            *[(name, param.shape) for name, param in m.named_parameters()][0]
        )
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
print('3' * 100)
print(net[0].weight.data)

print('4' * 100)
shared = nn.Linear(8, 8)
net = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    shared,
    nn.ReLU(),
    shared,
    nn.ReLU(),
    nn.Linear(8, 1)
)
net(X)

print('5' * 100)
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
print(net[2].weight.data[0] == net[4].weight.data[0])
