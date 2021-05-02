import torch

import torchtext

print('GPU:', torch.cuda.is_available())
torch.manual_seed(123)
TEXT = torchtext.legacy.data.Field(tokenize='spacy')
LABEL = torchtext.legacy.data.LabelField(dtype=torch.float)
train_data, test_data = torchtext.legacy.datasets.IMDB.splits(TEXT, LABEL)

print('len of train data:', len(train_data))
print('len of test data:', len(test_data))
print(train_data.examples[15].text)
print(train_data.examples[15].label)

# word2vec, glove

TEXT.build_vocab(train_data, max_size=10000, vectors='glove.6B.100d')
LABEL.build_vocab(train_data)
