import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F

dtype = torch.FloatTensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good
embedding_size = 2
sequence_length = len(sentences[0])
num_classes = len(set(labels))
batch_size = 3
word_sequence = " ".join(sentences).split()
word_set = set(word_sequence)
word2vec = {c: i for i, c in enumerate(word_set)}
voc_size = len(word2vec)


def make_data(sentences, labels):
    inputs = []
    outputs = []
    for sen in sentences:
        inputs.append([word2vec[n] for n in sen.split()])
    targets = []
    for out in labels:
        targets.append(out)
    return inputs, targets


input_batch, target_batch = make_data(sentences, labels)
input_batch, target_batch = torch.LongTensor(input_batch), torch.LongTensor(target_batch)

dataset = torch.utils.data.TensorDataset(input_batch, target_batch)
loader = torch.utils.data.DataLoader(dataset, batch_size, True)
