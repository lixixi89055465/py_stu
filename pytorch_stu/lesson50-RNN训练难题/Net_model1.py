import torch
import numpy as np

num_time_steps = 50
input_size = 1
hidden_size = 16
output_size = 1
lr = 1e-2


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = torch.nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        for p in self.rnn.parameters():
            torch.nn.init.normal_(p, mean=0.0, std=0.001)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)
        out = out.view(-1, hidden_prev)
        out = self.linear(out)
        out = out.unsqueeze(dim=0)
        return out, hidden_prev


model = Net()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr)
hidden_prev = torch.zeros(1, 1, hidden_size)

for iter in range(6000):
    start = np.random.randint(3, size=1)[0]
    break

print(start)
