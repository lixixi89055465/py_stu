import torch
import numpy as np

sentences = (
    'GitHub Actions makes it easy to automate all your software workflows '
    'from continuous integration and delivery to issue triage and more'
)
word2idx = {w: i for i, w in enumerate(list(set(sentences.split())))}
idx2word = {i: w for i, w in enumerate(list(set(sentences.split())))}
n_class = len(word2idx)
max_len = len(sentences.split())
n_hidden = 5


def make_data(sentence):
    input_batch = []
    target_batch = []
    words = sentences.split()
    for i in range(max_len - 1):
        input = [word2idx[n] for n in words[:(i + 1)]]
        input = input + [0] * (max_len - len(input))
        target = word2idx[words[i + 1]]
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)
    return torch.Tensor(input_batch), torch.LongTensor(target_batch)


input_batch, target_batch = make_data(sentences)

dataset = torch.utils.data.TensorDataset(input_batch, target_batch)
loader = torch.utils.data.DataLoader(dataset, 16, True)


class BiLSTM(torch.nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=n_class, hidden_size=n_hidden,bidirectional=True)
        # fc
        self.fc = torch.nn.Linear(n_hidden * 2, n_class)

    def forward(self, X):
        batch_size = X.shape[0]
        input = X.transpose(0, 1)
        hidden_state=torch.randn(1*2,batch_size,n_hidden)
        cell_state=torch.randn(1*2,batch_size,n_hidden)
        outputs,(_,_)=self.lstm(input,(hidden_state,cell_state))
        outputs=outputs[-1]
        model=self.fc(outputs)
        return model

model=BiLSTM()
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)

for epoch in range(10000):
    for x,y in loader:
        pred=model(x)
        loss=criterion(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1)%100==0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

# pred
predict=model(input_batch).data.max(1,keepdim=True)[1]
print(sentences)

print([idx2word[n.item()] for n in predict.squeeze()])
