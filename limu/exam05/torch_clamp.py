# -*- coding: utf-8 -*-
# @Time    : 2023/12/16 17:01
# @Author  : nanji
# @Site    : 
# @File    : torch_clamp.py
# @Software: PyCharm 
# @Comment :
import torch

boxes_nms = torch.randint(-100, 1000, (3, 4))
print(boxes_nms)
boxes_nms[:, 0] = torch.clamp(boxes_nms[:, 0], min=0)
boxes_nms[:, 1] = torch.clamp(boxes_nms[:, 1], min=0)
boxes_nms[:, 2] = torch.clamp(boxes_nms[:, 2], max=448)
boxes_nms[:, 3] = torch.clamp(boxes_nms[:, 3], max=448)
print('0'*100)
print(boxes_nms)
