Prediction = []
for seed in range(106, 131):
    print('Seed:', seed)
    import numpy as np

    print('Loading images ...')
    data_root = '../images/timit_11/'
    train = np.load(data_root + 'train_11.npy')
    train_label = np.load(data_root + 'train_label_11.npy')
    test = np.load(data_root + 'test_11.npy')
    print('Size of training images: {}'.format(train.shape))
    print('Size of testing images: {}'.format(test.shape))

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


    VAL_RATIO = 0.0
    percent = int(train.shape[0] * (1 - VAL_RATIO))
    train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
    print('Size of training set: {}'.format(train_x.shape))
    print('Size of validation set: {}'.format(val_x.shape))

    BATCH_SIZE = 3000
    from torch.utils.data import DataLoader

    train_set = TIMITDataset(train_x, train_y)
    val_set = TIMITDataset(val_x, val_y)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # only shuffle the training images
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    import gc

    del train, train_label, train_x, train_y, val_x, val_y
    gc.collect()

    import torch
    import torch.nn as nn


    class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            self.layer1 = nn.Linear(429, 1024)
            self.layer2 = nn.Linear(1024, 1024)
            self.layer3 = nn.Linear(1024, 1024)
            self.layer4 = nn.Linear(1024, 1024)
            self.layer5 = nn.Linear(1024, 1024)
            self.out = nn.Linear(1024, 39)
            self.bn1 = nn.BatchNorm1d(1024)
            self.bn2 = nn.BatchNorm1d(1024)
            self.bn3 = nn.BatchNorm1d(1024)
            self.bn4 = nn.BatchNorm1d(1024)
            self.bn5 = nn.BatchNorm1d(1024)
            self.dropout1 = nn.Dropout(p=0.3)
            self.dropout2 = nn.Dropout(p=0.3)
            self.dropout3 = nn.Dropout(p=0.3)
            self.dropout4 = nn.Dropout(p=0.3)
            self.dropout5 = nn.Dropout(p=0.3)
            self.act_fn = nn.Sigmoid()

        def forward(self, x):
            x = self.layer1(x)
            x = self.bn1(x)
            x = self.act_fn(x)
            x = self.dropout1(x)
            x = self.layer2(x)
            x = self.bn2(x)
            x = self.act_fn(x)
            x = self.dropout2(x)
            x = self.layer3(x)
            x = self.bn3(x)
            x = self.act_fn(x)
            x = self.dropout3(x)
            x = self.layer4(x)
            x = self.bn4(x)
            x = self.act_fn(x)
            x = self.dropout4(x)
            x = self.layer5(x)
            x = self.bn5(x)
            x = self.act_fn(x)
            x = self.dropout5(x)
            x = self.out(x)
            return x


    # check device
    def get_device():
        return 'cuda' if torch.cuda.is_available() else 'cpu'


    # fix random seed
    def same_seeds(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True


    same_seeds(seed)
    device = get_device()
    num_epoch = 200
    learning_rate = 0.001
    model_path = './model.ckpt'
    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # start training
    best_acc = 0.0
    for epoch in range(num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        # training
        model.train()  # set the model to training mode
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
            batch_loss.backward()
            optimizer.step()

            train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
            train_loss += batch_loss.item()
        # validation
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
    # if not validating, save the last epoch
    if len(val_set) == 0:
        torch.save(model.state_dict(), model_path)
        print('saving model at last epoch')

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
    for i in range(3, len(predict) - 3):
        if predict[i] != predict[i - 1] and predict[i] != predict[i + 1]:
            if np.argmax(np.bincount(np.array(predict[i - 3:i]))) == np.argmax(
                    np.bincount(np.array(predict[i + 1:i + 4]))):
                predict[i] = np.argmax(np.bincount(np.array(predict[i - 3:i])))
    Prediction.append(predict)

for i in range(len(predict)):
    predict[i] = np.argmax(np.bincount(np.array([Prediction[0][i], Prediction[1][i], Prediction[2][i], Prediction[3][i], Prediction[4][i],
                                                 Prediction[5][i], Prediction[6][i], Prediction[7][i], Prediction[8][i], Prediction[9][i],
                                                 Prediction[10][i], Prediction[11][i], Prediction[12][i], Prediction[13][i], Prediction[14][i],
                                                 Prediction[15][i], Prediction[16][i], Prediction[17][i], Prediction[18][i], Prediction[19][i],
                                                 Prediction[20][i], Prediction[21][i], Prediction[22][i], Prediction[23][i], Prediction[24][i]])))
with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))