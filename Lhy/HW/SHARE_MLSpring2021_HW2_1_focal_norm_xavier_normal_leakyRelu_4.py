# !gdown --id '1HPkcmQmFGu-3OknddKIa5dNDsR05lIQR' --output data.zip
# !unzip data.zip
# !ls
import numpy as np

data_root = '../data/timit_11/'
train = np.load(data_root + 'train_11.npy')
train_label = np.load(data_root + 'train_label_11.npy')
test = np.load(data_root + 'test_11.npy')
print(train.shape)
print(test.shape)

# create Dataset
import torch
from torch.utils.data import Dataset


class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


VAL_RATIO = 0.1
percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
BATCH_SIZE = 64
from torch.utils.data import DataLoader

train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)
# train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
# val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
# it=iter(train_set)
# a=next(it)

import gc

del train, train_label, train_x, train_y, val_x, val_y
gc.collect()

# create model
import torch
import torch.nn as nn


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = torch.nn.Linear(429, 1024)
        self.layer2 = torch.nn.Linear(1024, 1024)
        self.layer2_5 = torch.nn.Linear(1024, 512)
        self.layer3 = torch.nn.Linear(512, 128)
        self.out = torch.nn.Linear(128, 39)
        self.act_fn = torch.nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act_fn(x)

        x = self.layer2(x)
        x = self.act_fn(x)

        x = self.layer2_5(x)
        x = self.act_fn(x)

        x = self.layer3(x)
        x = self.act_fn(x)

        x = self.out(x)
        # x = self.act_fn(x)
        return x


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# fix random seed for reproducibility
same_seeds(0)
device = get_device()
print(device)
print(f'DEVICE:{device}')
num_epoch = 20
learning_rate = 0.0001
# the path were checkpoint saved
model_path = './model.ckpt'
model = Classifier().to(device)

model.apply(weight_init)

optimizer1 = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
optimizer2 = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0001)
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

criterion = nn.CrossEntropyLoss().to(device)


def scheduler(epoch):
    if epoch < num_epoch * 0.3:
        return learning_rate
    if epoch < num_epoch * 0.8:
        return learning_rate * 0.1
    return learning_rate * 0.01


def loss1(inputs, targets, alpha=0.75, gamma=2, size_average=True):
    logp = criterion(inputs, targets)
    p = torch.exp(-logp)
    loss = alpha * (1 - p) ** gamma * logp * targets.long() + \
           (1 - alpha) * (p) ** gamma * logp * (1 - targets.long())
    return torch.abs(loss.mean())


best_acc = 0.
for epoch in range(num_epoch):
    if epoch < num_epoch/2:
        optimizer = optimizer1
    else:
        optimizer = optimizer2

    train_acc = 0.
    train_loss = 0.
    val_acc = 0.
    val_loss = 0.
    # training
    model.train()  # set model to training mode
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # batch_loss = criterion(outputs, labels)
        batch_loss = loss1(outputs, labels)
        _, train_pred = torch.max(outputs, 1)
        # regularization_loss = 0.
        # for param in model.parameters():
        #     regularization_loss += torch.sum(torch.abs(param))
        # classify_loss = batch_loss + 0.01 * regularization_loss
        # classify_loss.backward()

        batch_loss.backward()
        optimizer.step()
        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()
    if len(val_set) > 0:
        model.eval()
        with torch.no_grad():
            for i in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)

                _, val_pred = torch.max(outputs, 1)
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                val_loss += batch_loss.item()
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader),
                val_acc / len(val_set), val_loss / len(val_loader)
            ))
            if len(val_set) > 0:
                val_acc = 0.
                val_loss = 0.
                model.eval()  # set the model to evaluation mode
                with torch.no_grad():
                    for i, data in enumerate(val_loader):
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        batch_loss = criterion(outputs, labels)
                        _, val_pred = torch.max(outputs, 1)

                        val_acc += (
                                val_pred.cpu() == labels.cpu()).sum().item()  # get the index of the class with the highest probability
                        val_loss += batch_loss.item()

                    print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                        epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader),
                        val_acc / len(val_set), val_loss / len(val_loader)
                    ))

                    # if the model improves, save a checkpoint at this epoch
                    if val_acc > best_acc:
                        best_acc = val_acc
                        torch.save(model.state_dict(), model_path)
                        print('saving model with acc {:.3f}'.format(best_acc / len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader)
        ))
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch ')

# test
# create testing dataset
test_set = TIMITDataset(test, None)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# create model and load weights from checkpoint
model = Classifier().to(device)
model.load_state_dict(torch.load(model_path))

predict = []
model.eval()  # set the model to evaluation mode
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, test_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability

        for y in test_pred.cpu().numpy():
            predict.append(y)

with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))
