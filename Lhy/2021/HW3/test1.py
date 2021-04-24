import torch
a=torch.randn(3,4) #随机生成一个shape（3，4）的tensort
b=torch.randn(2,4) #随机生成一个shape（2，4）的tensor

torch.cat([a, b], dim=0)
#返回一个shape（5，4）的tensor
#把a和b拼接成一个shape（5，4）的tensor，
#可理解为沿着行增加的方向（即纵向）拼接