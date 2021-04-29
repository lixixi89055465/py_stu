import torch
decoder_layer=torch.nn.TransformerDecoderLayer(d_model=512,nhead=8)
print(decoder_layer)
memory=torch.rand(10,32,512)
tgt=torch.rand(20,32,512)
out=decoder_layer(tgt,memory)
print(out.shape)
