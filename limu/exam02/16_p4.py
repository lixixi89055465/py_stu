import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load('x-file')
print('0' * 100)
print(x2)

y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')
print('1' * 100)
print(x2, y2)

mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print('2' * 100)
print(mydict2)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
print('3' * 100)
print(Y)

torch.save(net.state_dict(), 'mlp.params')
print('4' * 100)
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print('5' * 100)
clone.eval()
print('6' * 100)
Y_clone = clone(X)
print(Y_clone == Y)
