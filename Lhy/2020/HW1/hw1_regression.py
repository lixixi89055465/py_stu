import sys
import pandas as pd
import numpy as np

# !gdown --id '1wNKAxQ29G15kgpBy_asjTcZRRgmsCZRm' --output images.zip
# !unzip images.zip
data = pd.read_csv('../images/hw1_data/train.csv', encoding='big5')
print(data.shape)
print(data[:1])
data = data.iloc[:, 3:]
print(data.shape)
print('1' * 20)
print(type(data))
print((data == 'NR').shape)
data[data == 'NR'] = 0
raw_data = data.to_numpy()
print(type(raw_data))
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24:(day + 1) * 24] = \
            raw_data[18 * (20 * month + day):18 * (20 * month + day + 1), :]
    month_data[month] = sample

x = np.empty([12 * 471, 18 * 9], dtype=float)
y = np.empty([12 * 471, 1], dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour:day * 24 + hour + 9].reshape(1,
                                                                                                                    -1)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]

mean_x = np.mean(x, axis=0)
std_x = np.std(x, axis=0)

# Normalize 1

for i in range(len(x)):
    for j in range(len(x[0])):
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

import math

x_train_set = x[:math.floor(len(x) * 0.8), :]
y_train_set = y[:math.floor(len(y) * 0.8), :]

x_validation = x[math.floor(len(x) * 0.8):, :]
y_validation = y[math.floor(len(y) * 0.8):, :]

print(x_train_set.shape)
print(y_train_set.shape)
print(x_validation.shape)
print(y_validation.shape)
print('2' * 30)
dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float)
print(w.shape)
print(x.shape)
learning_rate = 100
iter_time = 1000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
print(y.shape)
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12)
    # if t % 100 == 0:
    #     print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
np.save('../images/weight.npy', w)
# print(w)


# testdata = pd.read_csv('gdrive/My Drive/hw1-regression/test.csv', header = None, encoding = 'big5')
testdata = pd.read_csv('../images/hw1_data/test.csv', header=None, encoding='big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18 * 9], dtype=float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18 * (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)
print(test_x)

w = np.load('../images/weight.npy')
print(w.shape)
ans_y = np.dot(test_x, w)
print(test_x.shape)
print(ans_y.shape)
print(ans_y)

import csv
with open('../images/submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)