import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# matplotlib inline
import os

path = 'LogiReg_data.txt'
pdData = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
print(pdData.head())

positive = pdData[pdData['Admitted'] == 1]
negative = pdData[pdData['Admitted'] == 0]
print(positive.head())


# fig,ax=plt.subplots(figsize=(10,5))
# ax.scatter(positive['Exam 1'],positive['Exam 2'],s=50,c='b',marker='o',label='Admitted')
# ax.scatter(negative['Exam 1'],negative['Exam 2'],s=30,c='r',marker='x',label='Not Admitted')
# ax.legend()
# ax.set_xlabel('Exam 1 Score')
# ax.set_ylabel('Exam 2 Score')
# plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# nums=np.arange(start=-10,stop=10,step=.1)
# fig,ax=plt.subplots(figsize=(12,4))
# ax.plot(nums,sigmoid(nums),'r')
# plt.show()


def model(X, theta):
    return sigmoid(np.dot(X, theta.T))


pdData.insert(0, 'Ones', 1)
print(pdData.head())
orig_data = pdData.values
cols = orig_data.shape[1]
X = orig_data[:, :cols - 1]
y = orig_data[:, cols - 1:cols]
# print('1'*100)
# print(X.shape)
# print(y.shape)
# print(X[:5])
# print(y[:5])
theta = np.zeros([1, 3])


def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply((1 - y), np.log(1 - model(X, theta)))
    return np.sum((left - right) / len(X))


# cost(X, y, theta)
def gradient(X1, y1, theta):
    grad = np.zeros(theta.shape)
    error = (model(X1, theta) - y1).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error, X1[:, j])
        grad[0, j] = np.sum(term) / len(X1)
    return grad


STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2


def stopGriterion(type, value, threshold):
    # 设定三种不同的停止策略
    if type == STOP_ITER:
        return value > threshold
    elif type == STOP_COST:
        return abs(value[-1] - value[-2]) < threshold
    elif type == STOP_GRAD:
        return np.linalg.norm(value) < threshold


def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols - 1]
    y = data[:, cols - 1:]
    return X, y


import time

n = 100


def descent(data, theta, batchSize, stopType, thresh, alpha):
    # 梯度下降求解
    init_time = time.time()
    i = 0  # 迭代次数
    k = 0  # batch
    X, y = shuffleData(data)
    grad = np.zeros(theta.shape)
    costs = [cost(X, y, theta)]
    while True:
        grad = gradient(X[k:k + batchSize], y[k:k + batchSize], theta)
        k += batchSize  # 取batch数量个数据
        if k >= n:
            k = 0
            X, y = shuffleData(data)
        theta = theta - alpha * grad
        tempCost = cost(X, y, theta)
        costs.append(tempCost)
        i += 1
        if stopType == STOP_ITER:
            value = i
        elif stopType == STOP_COST:
            value = costs
        elif stopType == STOP_GRAD:
            value = grad
        if stopGriterion(stopType, value, thresh): break
    return theta, i - 1, costs, grad, time.time() - init_time


def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    name = 'Original' if (data[:, 1] > 2).sum() > 1 else 'Scaled'
    name += '  data -learning rate:{} - '.format(alpha)
    if batchSize == n:
        strDescType = 'Gradient'
    elif batchSize == 1:
        strDescType = 'Stochastic'
    else:
        strDescType = 'Mini - batch {}'.format(batchSize)
    name += strDescType + 'descent = stop: '
    if stopType == STOP_ITER:
        strStop = '{} iterations  '.format(thresh)
    elif stopType == STOP_COST:
        strStop = 'costs change < {}'.format(thresh)
    else:
        strStop = ' gradient norm < {}'.format(thresh)
    name += strStop
    print('*** {} \n Theta : {} - Iter:{}- last cost:{:03.2f} - Duration :{:03.2f}s'
          .format(name,theta, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' -   Error vs . Iteration   ')
    plt.show()
    return theta


n = 100
# runExpe(orig_data, theta, n, STOP_ITER, thresh=50000, alpha=0.00001)
runExpe(orig_data, theta, n, STOP_COST, thresh=0.000001, alpha=0.001)
