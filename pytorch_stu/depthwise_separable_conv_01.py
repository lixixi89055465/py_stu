import torch
import torch.nn as nn
from torchsummary import summary



class CRNN(nn.Module):
    def __init__(self,in_channel,out_channel,group):
        super(CRNN, self).__init__()
        self.conv=nn.Conv2d(in_channels=in_channel,
                            out_channels=out_channel,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            groups=group,
                            bias=False )
    def forward(self, input):
        out=self.conv(input)
        return out
class DEPTHWISECONV(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DEPTHWISECONV,self).__init__()
        self.depth_conv=nn.Conv2d(in_channels=in_ch,
                                  out_channels=in_ch,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  groups=in_ch
                                  )
        self.point_conv=nn.Conv2d(in_channels=in_ch,
                                  out_channels=out_ch,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  groups=1)
    def forward(self, input):
        out=self.depth_conv(input)
        out=self.point_conv(out)
        return out

if __name__ == '__main__':
    device=torch.device('cuda' if torch.cuda.is_available()else 'cpu')
    conv=CRNN(3,6,1).to(device)
    print(summary(conv,input_size=(3,32,32)))
    dp=DEPTHWISECONV(3,6).to(device)
    print(summary(dp,input_size=(3,32,32)))


