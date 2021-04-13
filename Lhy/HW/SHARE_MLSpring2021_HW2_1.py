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


VAL_RATIO = 0.2
percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
BATCH_SIZE = 128
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


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = torch.nn.Linear(429, 1024)
        self.layer2 = torch.nn.Linear(1024, 512)
        self.layer3 = torch.nn.Linear(512, 128)
        self.out = torch.nn.Linear(128, 39)
        self.act_fn = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.act_fn(x)

        x = self.layer2(x)
        x = self.act_fn(x)

        x = self.layer3(x)
        x = self.act_fn(x)

        x = self.out(x)
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
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_acc = 0.
for epoch in range(num_epoch):
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
        batch_loss = criterion(outputs, labels)
        _, train_pred = torch.max(outputs, 1)
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
