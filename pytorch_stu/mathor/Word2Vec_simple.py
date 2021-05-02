'''
  code by Tae Hwan Jung(Jeff Jung) @graykode, modify by wmathor
  6/11/2020
'''
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data as Data

dtype = torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentences = ["jack like dog", "jack like cat", "jack like animal",
             "dog cat animal", "banana apple cat dog like", "dog fish milk like",
             "dog cat animal like", "jack like apple", "apple like", "jack like banana",
             "apple banana jack movie book music like", "cat dog hate", "cat dog like"]

word_sequence = " ".join(sentences).split()
print('word_sequence:')
print(word_sequence)
vocab = list(set(word_sequence))
print('vocab:')
print(vocab)
word2idx = {w: i for i, w in enumerate(vocab)}
print('word2idx:')
print(word2idx)
# Word2Vec Parameters
batch_size = 8
embedding_size = 2  # 2 dim vector represent one word
C = 2  # window size
voc_size = len(vocab)
# data preprocess
skip_grams = []
# for idx in range(C,len(word_sequence)-C):
#     print(idx)
for idx in range(C, len(word_sequence) - C):
    center = word2idx[word_sequence[idx]]
    centext_idx = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))
    context = [word2idx[word_sequence[i]] for i in centext_idx]
    for w in context:
        skip_grams.append([center, w])


# 2.
def make_data(skip_grams):
    input_data = []
    output_data = []
    for i in range(len(skip_grams)):
        input_data.append(np.eye(voc_size)[skip_grams[i][0]])
        output_data.append(skip_grams[i][1])
    return input_data, output_data


# 3.
input_data, output_data = make_data(skip_grams)
input_data, output_data = torch.Tensor(input_data), torch.LongTensor(output_data)
dataset = torch.utils.data.TensorDataset(input_data, output_data)
loader = torch.utils.data.DataLoader(dataset, batch_size, True)


# Model
class Word2Vec(torch.nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(voc_size, embedding_size).type(dtype))
        self.V = torch.nn.Parameter(torch.randn(embedding_size, voc_size).type(dtype))

    def forward(self, X):
        hidden_layer = torch.matmul(X, self.W)
        output_layer = torch.matmul(hidden_layer, self.V)
        return output_layer


model = Word2Vec().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(2000):
    for i, (batch_x, batch_y) in enumerate(loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(epoch + 1, loss.item())

for i, label in enumerate(vocab):
    W, WT = model.parameters()
    x, y = float(W[i][0]), float(W[i][1])
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()
