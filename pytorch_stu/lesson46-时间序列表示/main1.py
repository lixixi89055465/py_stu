import torch
from torchnlp import word_to_vector

word_to_vector = {'hello': 0, 'word': 1}

lookup_tensor = torch.tensor([word_to_vector['hello']], dtype=torch.long)

print(lookup_tensor)
helloembeds = torch.nn.Embedding(2, 5)
print(helloembeds)


