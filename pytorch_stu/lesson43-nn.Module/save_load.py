import torch

device = torch.device('cuda')
net = torch.nn.Sequential(
    torch.nn.Conv2d(1, 32, 5, 1, 1),
    torch.nn.MaxPool2d(2, 2),
    torch.nn.ReLU(True),
    torch.nn.BatchNorm2d(32),

    torch.nn.Conv2d(32, 64, 3, 1, 1),
    torch.nn.ReLU(True),
    torch.nn.BatchNorm2d(64),

    torch.nn.Conv2d(64, 64, 3, 1, 1),
    torch.nn.MaxPool2d(2, 2),
    torch.nn.ReLU(True),
    torch.nn.BatchNorm2d(64),

    torch.nn.Conv2d(64, 128, 3, 1, 1),
    torch.nn.ReLU(True),
    torch.nn.BatchNorm2d(128)
)

net.to(device)
net.load_state_dict(torch.load('ckpt.mdl'))
# train
torch.save(net.state_dict(), "ckpt.mdl")


# train 状态
net.train()

# test 状态
net.eval() 
