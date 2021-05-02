import torch


class ResBlk(torch.nn.Module):
    def __init__(self, ch_in, ch_out):
        self.conv1 = torch.nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(ch_out)
        self.conv2 = torch.nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(ch_out)
        self.extra = torch.nn.Sequential()
        if ch_out != ch_in:
            self.extra = torch.nn.Sequential(
                torch.nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=1),
                torch.nn.BatchNorm1d(ch_out)
            )

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        return out
