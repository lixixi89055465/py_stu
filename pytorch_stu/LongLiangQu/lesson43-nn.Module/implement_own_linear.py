import torch


class MyLinear(torch.nn.Module):
    def __init__(self, inp, outp):
        super(MyLinear, self).__init__()
        self.w = torch.nn.Parameter(torch.randn(outp, inp))
        self.b = torch.nn.Parameter(torch.randn(outp))

    def forward(self, x):
        x = x @ self.w.t() + self.b
        return x
