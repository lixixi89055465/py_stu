import torch
import torchvision

batch_size = 200
learning_rate = 0.01
epochs = 10

train_db = torchvision.datasets.MNIST('../data', train=True, download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.1317,), (0.2081))
                                      ]))

train_loader = torch.utils.data.DataLoader(
    train_db,
    batch_size=batch_size, shuffle=True
)

test_db = torchvision.datasets.MNIST('../data', train=False, transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
]))

test_loader = torch.utils.data.DataLoader(test_db, batch_size=batch_size, shuffle=True)

print('train:', len(train_db), 'test:', len(test_db))
train_db, val_db = torch.utils.data.random_split(train_db, [50000, 10000])
print('db1:', len(train_db), 'db2:', len(val_db))

train_loader = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(val_db, batch_size, shuffle=True)


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.module = torch.nn.Sequential(
            torch.Linear(784, 200),
            torch.nn.LeakyReLU(inplace=True),
            torch.Linear(200, 200),
            torch.nn.LeakyReLU(inplace=True),
            torch.Linear(200, 10),
            torch.nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)
        return x


device = torch.device('cuda:0')
net = MLP().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
criteon = torch.nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()
        logits = net(data)
        loss = criteon(logits, target)
        optimizer.zero_grad()
