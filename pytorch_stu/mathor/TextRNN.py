import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def make_batch():
    input_batch = []
    target_batch = []
    for sen in sentences:
        word = sen.split()


class TextRNN(torch.nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        self.rnn = torch.nn.RNN(input_size=n_class, hidden_size=n_hidden)
        self.W = torch.nn.Linear(n_hidden, n_class, bias=False)
        self.b = torch.nn.Parameter(torch.ones([n_class]))

    def forward(self, hidden, X):
        X = X.transpose(0, 1)
        outputs, hidden = self.rnn(X, hidden)
        outputs = output[-1]
        model = self.W(outputs) + self.b
        return model


if __name__ == '__main__':
    n_step = 2
    n_hidden = 5
    sentences = ["i like dog", "i love coffee", "i hate milk"]
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_dict)
    batch_size = len(sentences)
    model = TextRNN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    input_batch, target_batch = make_batch()
    input_batch = torch.FloatTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)
    # Training
    for epoch in range(5000):
        optimizer.zero_grad()
        hidden = torch.zeros(1, batch_size, n_hidden)
        output = model(hidden, input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()
    input = [sen.split()[:2] for sen in sentences]
    hidden = torch.zeros(1, batch_size, n_hidden)
    predict = model(hidden, input_batch).data.max(1, keepdim=True)[1]
    print([sen.split()[:2] for sen in sentences],'->',[number_dict[n.item()]for n in predict.squeeze()])

