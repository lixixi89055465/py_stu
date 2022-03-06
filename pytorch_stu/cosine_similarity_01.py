'''

'''
import torch
import torch.nn.functional as F
input1=torch.randn(100,128)
input2=torch.randn(100,128)
output=F.cosine_similarity(input1,input2)
print(output.shape)

