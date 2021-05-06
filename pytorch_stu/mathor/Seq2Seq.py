import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

letter = [c for c in 'SE?abcdefghijklmnopqrstuvwxyz']
print(letter)

letter2idx = {n: i for i, n in enumerate(letter)}
print(letter2idx)
seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]
# seq2seq parameter

n_step = max([max(len(i), len(j)) for i, j in seq_data])
print(n_step)
n_hidden = 128
n_class = len(letter2idx)
batch_size = 3
print(n_class)


def make_data(seq_data):
    enc_input_all, dec_input_all, dec_output_all = [], [], []
    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + '?' * (n_step - len(seq[i]))
        enc_input = [letter2idx[n] for n in (seq[0] + 'E')]
        dec_input = [letter2idx[n] for n in ('S' + seq[1])]
        dec_output = [letter2idx[n] for n in (seq[1] + 'E')]
        enc_input_all.append(np.eye(n_class)[enc_input])
        dec_input_all.append(np.eye(n_class)[dec_input])
        dec_output_all.append(dec_output)
    return torch.Tensor(enc_input_all), torch.Tensor(dec_input_all), torch.LongTensor(dec_output_all)


enc_input_all, dec_input_all, dec_output_all = make_data(seq_data)
print(enc_input_all.shape)
print(dec_input_all.shape)
print(dec_output_all.shape)


class TranslateDataset(torch.utils.data.Dataset):
    def __init__(self, enc_input_all, dec_input_all, dec_output_all):
        self.enc_input_all = enc_input_all
        self.dec_input_all = dec_input_all
        self.dec_output_all = dec_output_all

    def __len__(self):
        return len(self.enc_input_all)

    def __getitem__(self, idx):
        return self.enc_input_all[idx], self.dec_input_all[idx], self.dec_output_all[idx]


loader = torch.utils.data.DataLoader(TranslateDataset(enc_input_all, dec_input_all, dec_output_all),
                                     batch_size, True)


# loader = Data.DataLoader(TranslateDataset(enc_input_all, dec_input_all, dec_output_all), batch_size, True)


# Model
class Seq2Seq(torch.nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = torch.nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.decoder = torch.nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.fc = torch.nn.Linear(n_hidden, n_class)

    def forward(self, enc_input, enc_hidden, dec_input):
        enc_input = enc_input.transpose(0, 1)
        dec_input = dec_input.transpose(0, 1)
        _, h_t = self.encoder(enc_input, enc_hidden)
        outputs, _ = self.decoder(dec_input, h_t)
        model = self.fc(outputs)
        return model


model = Seq2Seq().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5000):
    for enc_input_batch, dec_input_batch, dec_output_batch in loader:
        h_0 = torch.zeros(1, batch_size, n_hidden).to(device)
        (enc_input_batch, dec_input_batch, dec_output_batch) = (
            enc_input_batch.to(device), dec_input_batch.to(device), dec_output_batch.to(device))
        pred = model(enc_input_batch, h_0, dec_input_batch)
        pred = pred.transpose(0, 1)  # [batch_size, n_step+1(=6), n_class]
        loss = 0.
        for i in range(len(dec_output_batch)):
            pred_i = pred[i]
            batch_i = dec_output_batch[i]
            loss += criterion(pred[i], dec_output_batch[i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print('Epoch :', '%04d' % (epoch + 1), 'cost = ', '{:.6f}'.format(loss))


# Test
def translate(word):
    enc_input, dec_input, _ = make_data([[word, '?' * n_step]])
    enc_input, dec_input = enc_input.to(device), dec_input.to(device)
    # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
    hidden = torch.zeros(1, 1, n_hidden).to(device)
    output = model(enc_input, hidden, dec_input)
    # output : [n_step+1, batch_size, n_class]

    predict = output.data.max(2, keepdim=True)[1]  # select n_class dimension
    decoded = [letter[i] for i in predict]
    translated = ''.join(decoded[:decoded.index('E')])

    return translated.replace('?', '')


print('test')
print('man ->', translate('man'))
print('mans ->', translate('mans'))
print('king ->', translate('king'))
print('black ->', translate('black'))
print('up ->', translate('up'))
