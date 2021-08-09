import torch
# from torch import nn, optim
import torch
# from torchtext import images, datasets
import torchtext

# from torchtext.legacy.images import Field

print('GPU:', torch.cuda.is_available())
torch.manual_seed(123)

TEXT = torchtext.legacy.data.Field(tokenize='spacy')
LABEL = torchtext.legacy.data.LabelField(dtype=torch.float)
train_data, test_data = torchtext.legacy.datasets.IMDB.splits(TEXT, LABEL)
