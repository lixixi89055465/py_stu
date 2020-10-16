# Train a linear Classifier

import numpy as np

import matplotlib.pyplot as plt

np.random.seed(10)
N = 100
D = 2
K = 3
X = np.zeros((N * K, D))

y = np.zeros(N * K, dtype='uint8')
for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j

W = 0.01 * np.random.randn(D, K)

b = np.zeros((1, K))

step_size = 1e-0
reg = 1e-3

num_examples = X.shape[0]
for i in range(1000):
    scores = np.dot(X, W) + b
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs) / num_examples
    reg_loss = 0.5 * reg * np.sum(W * W)
    loss = data_loss + reg_loss
    if i % 100 == 0:
        print(' iteration %d : loss  %f' % (i, loss))
