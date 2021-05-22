# -*- coding: utf-8 -*-
"""“HW04.ipynb”的副本

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yok_Ivx0nUXsMxiaSpGLFHXqb8qR3Zbl

# Task description
- Classify the speakers of given features.
- Main goal: Learn how to use transformer.
- Baselines:
  - Easy: Run sample code and know how to use transformer.
  - Medium: Know how to adjust parameters of transformer.
  - Hard: Construct [conformer](https://arxiv.org/abs/2005.08100) which is a variety of transformer. 

- Other links
  - Kaggle: [link](https://www.kaggle.com/t/859c9ca9ede14fdea841be627c412322)
  - Slide: [link](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/hw/HW04/HW04.pdf)
  - Data: [link](https://drive.google.com/file/d/1T0RPnu-Sg5eIPwQPfYysipfcz81MnsYe/view?usp=sharing)
  - Video (Chinese): [link](https://www.youtube.com/watch?v=EPerg2UnGaI)
  - Video (English): [link](https://www.youtube.com/watch?v=Gpz6AUvCak0)
  - Solution for downloading dataset fail.: [link](https://drive.google.com/drive/folders/13T0Pa_WGgQxNkqZk781qhc5T9-zfh19e?usp=sharing)
"""

from google.colab import drive
drive.mount('/content/drive')

!cp ./drive/MyDrive/Dataset.zip ./Datase.zip

!unzip Dataset.zip

"""# Download dataset
- **If all download links fail**
- **Please follow [here](https://drive.google.com/drive/folders/13T0Pa_WGgQxNkqZk781qhc5T9-zfh19e?usp=sharing)**
- **Data is [here](https://drive.google.com/file/d/1T0RPnu-Sg5eIPwQPfYysipfcz81MnsYe/view?usp=sharing)**
"""

"""
    For Google drive, You can download data form any link below.
    If a link fails, please use another one.
"""
""" Download link 1 of Google drive """
# !gdown --id '1T0RPnu-Sg5eIPwQPfYysipfcz81MnsYe' --output Dataset.zip
""" Download link 2 of Google drive """
# !gdown --id '1CtHZhJ-mTpNsO-MqvAPIi4Yrt3oSBXYV' --output Dataset.zip
""" Download link 3 of Google drive """
# !gdown --id '14hmoMgB1fe6v50biIceKyndyeYABGrRq' --output Dataset.zip
""" Download link 4 of Google drive """
# !gdown --id '1e9x-Pjl3n7-9tK9LS_WjiMo2lru4UBH9' --output Dataset.zip
""" Download link 5 of Google drive """
# !gdown --id '10TC0g46bcAz_jkiMl65zNmwttT4RiRgY' --output Dataset.zip
""" Download link 6 of Google drive """
# !gdown --id '1MUGBvG_JjqO0C2JYHuyV3B0lvaf1kWIm' --output Dataset.zip
""" Download link 7 of Google drive """
# !gdown --id '18M91P5DHwILNyOlssZ57AiPOR0OwutOM' --output Dataset.zip
""" For all download links fail, Please paste link into 'Paste link here' """
!gdown --id '1jNPtDZXHuJtPZ5XI636V0U87FugLUiHo' --output Dataset.zip
""" For Google drive, you can unzip the data by the command below. """
!unzip Dataset.zip

"""
    For Dropbox, we split dataset into five files. 
    Please download all of them.
"""
# If Dropbox is not work. Please use google drive.
# !wget https://www.dropbox.com/s/vw324newiku0sz0/Dataset.tar.gz.aa?dl=0
# !wget https://www.dropbox.com/s/z840g69e7lnkayo/Dataset.tar.gz.ab?dl=0
# !wget https://www.dropbox.com/s/hl081e1ggonio81/Dataset.tar.gz.ac?dl=0
# !wget https://www.dropbox.com/s/fh3zd8ow668c4th/Dataset.tar.gz.ad?dl=0
# !wget https://www.dropbox.com/s/ydzygoy2pv6gw9d/Dataset.tar.gz.ae?dl=0
# !cat Dataset.tar.gz.* | tar zxvf -

"""
    For Onedrive, we split dataset into five files. 
    Please download all of them.
"""
# !wget --no-check-certificate "https://onedrive.live.com/download?cid=10C95EE5FD151BFB&resid=10C95EE5FD151BFB%21106&authkey=ACB6opQR3CG9kmc" -O Dataset.tar.gz.aa
# !wget --no-check-certificate "https://onedrive.live.com/download?cid=93DDDDD552E145DB&resid=93DDDDD552E145DB%21106&authkey=AP6EepjxSdvyV6Y" -O Dataset.tar.gz.ab
# !wget --no-check-certificate "https://onedrive.live.com/download?cid=644545816461BCCC&resid=644545816461BCCC%21106&authkey=ALiefB0kI7Epb0Q" -O Dataset.tar.gz.ac
# !wget --no-check-certificate "https://onedrive.live.com/download?cid=77CEBB3C3C512821&resid=77CEBB3C3C512821%21106&authkey=AAXCx4TTDYC0yjM" -O Dataset.tar.gz.ad
# !wget --no-check-certificate "https://onedrive.live.com/download?cid=383D0E0146A11B02&resid=383D0E0146A11B02%21106&authkey=ALwVc4StVbig6QI" -O Dataset.tar.gz.ae
# !cat Dataset.tar.gz.* | tar zxvf -

"""# Data

## Dataset
- Original dataset is [Voxceleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/).
- The [license](https://creativecommons.org/licenses/by/4.0/) and [complete version](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/files/license.txt) of Voxceleb1.
- We randomly select 600 speakers from Voxceleb1.
- Then preprocess the raw waveforms into mel-spectrograms.

- Args:
  - data_dir: The path to the data directory.
  - metadata_path: The path to the metadata.
  - segment_len: The length of audio segment for training. 
- The architecture of data directory \\
  - data directory \\
  |---- metadata.json \\
  |---- testdata.json \\
  |---- mapping.json \\
  |---- uttr-{random string}.pt \\

- The information in metadata
  - "n_mels": The dimention of mel-spectrogram.
  - "speakers": A dictionary. 
    - Key: speaker ids.
    - value: "feature_path" and "mel_len"


For efficiency, we segment the mel-spectrograms into segments in the traing step.
"""

import os
import json
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
 
np.random.seed(10)
class myDataset(Dataset):
  def __init__(self, data_dir, segment_len=128):
    self.data_dir = data_dir
    self.segment_len = segment_len
 
    # Load the mapping from speaker neme to their corresponding id. 
    mapping_path = Path(data_dir) / "mapping.json"
    mapping = json.load(mapping_path.open())
    self.speaker2id = mapping["speaker2id"]
 
    # Load metadata of training data.
    metadata_path = Path(data_dir) / "metadata.json"
    metadata = json.load(open(metadata_path))["speakers"]
 
    # Get the total number of speaker.
    self.speaker_num = len(metadata.keys())
    self.data = []
    for speaker in metadata.keys():
      for utterances in metadata[speaker]:
        self.data.append([utterances["feature_path"], self.speaker2id[speaker]])
 
  def __len__(self):
    return len(self.data)
 
  def __getitem__(self, index):
    feat_path, speaker = self.data[index]
    # Load preprocessed mel-spectrogram.
    mel = torch.load(os.path.join(self.data_dir, feat_path))
 
    # Segmemt mel-spectrogram into "segment_len" frames.
    if len(mel) > self.segment_len:
      # Randomly get the starting point of the segment.
      start = random.randint(0, len(mel) - self.segment_len)
      # Get a segment with "segment_len" frames.
      mel = torch.FloatTensor(mel[start:start+self.segment_len])
    else:
      mel = torch.FloatTensor(mel)
    # Turn the speaker id into long for computing loss later.
    speaker = torch.FloatTensor([speaker]).long()
    return mel, speaker
 
  def get_speaker_number(self):
    return self.speaker_num

"""## Dataloader
- Split dataset into training dataset(90%) and validation dataset(10%).
- Create dataloader to iterate the data.

"""

import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence


def collate_batch(batch):
  # Process features within a batch.
  """Collate a batch of data."""
  mel, speaker = zip(*batch)
  # Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
  mel = pad_sequence(mel, batch_first=True, padding_value=-20)    # pad log 10^(-20) which is very small value.
  # mel: (batch size, length, 40)
  return mel, torch.FloatTensor(speaker).long()


def get_dataloader(data_dir, batch_size, n_workers):
  """Generate dataloader"""
  dataset = myDataset(data_dir)
  speaker_num = dataset.get_speaker_number()
  # Split dataset into training dataset and validation dataset
  trainlen = int(0.9 * len(dataset))
  lengths = [trainlen, len(dataset) - trainlen]
  trainset, validset = random_split(dataset, lengths)

  train_loader = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=n_workers,
    pin_memory=True,
    collate_fn=collate_batch,
  )
  valid_loader = DataLoader(
    validset,
    batch_size=batch_size,
    num_workers=n_workers,
    drop_last=True,
    pin_memory=True,
    collate_fn=collate_batch,
  )

  return train_loader, valid_loader, speaker_num

"""# Model
- TransformerEncoderLayer:
  - Base transformer encoder layer in [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  - Parameters:
    - d_model: the number of expected features of the input (required).

    - nhead: the number of heads of the multiheadattention models (required).

    - dim_feedforward: the dimension of the feedforward network model (default=2048).

    - dropout: the dropout value (default=0.1).

    - activation: the activation function of intermediate layer, relu or gelu (default=relu).

- TransformerEncoder:
  - TransformerEncoder is a stack of N transformer encoder layers
  - Parameters:
    - encoder_layer: an instance of the TransformerEncoderLayer() class (required).

    - num_layers: the number of sub-encoder-layers in the encoder (required).

    - norm: the layer normalization component (optional).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor

class PositionalEncoding(nn.Module):
  "Implement the PE function."
  def __init__(self, d_model, max_len=5000):
    super(PositionalEncoding, self).__init__()

    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                          -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + Variable(self.pe[:, :x.size(1)],
                      requires_grad=False)
    return x

class MHA(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", layer_norm_eps=1e-5):
        super(MHA, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(MHA, self).__setstate__(state)

    def forward(self, src: Tensor) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src)[0]
        return src2

class Conformer(nn.Module):
  def __init__(self, d_model=256):
    super().__init__()

    self.hidden = 512
    # in: (batch size, length, d_model)
    # out: (batch size, length, d_model)
    self.feedForward1 = nn.Sequential(
        # out: (length, batch size, d_model)
        nn.LayerNorm(d_model),
        # out: (length, batch size, d_model)
        nn.Linear(d_model, self.hidden), 
        nn.Hardswish(),
        # out: (length, batch size, d_model)
        nn.Dropout(0.4),
        # out: (length, batch size, d_model)
        nn.Linear(self.hidden, d_model), 
        # out: (length, batch size, d_model)
        nn.Dropout(0.4)
    )

    # in: (batch size, length, d_model)
    # out: (batch size, length, d_model)
    self.feedForward2 = nn.Sequential(
        # out: (length, batch size, d_model)
        nn.LayerNorm(d_model),
        # out: (length, batch size, d_model)
        nn.Linear(d_model, self.hidden), 
        nn.Hardswish(),
        # out: (length, batch size, d_model)
        nn.Dropout(0.4),
        # out: (length, batch size, d_model)
        nn.Linear(self.hidden, d_model), 
        # out: (length, batch size, d_model)
        nn.Dropout(0.4)
    )

    # in: (length, batch size, d_model)
    # out: (length, batch size, d_model)
    self.multiHeaded = nn.Sequential(
        nn.LayerNorm(d_model),
        PositionalEncoding(d_model),
        MHA(d_model, 8),
        nn.Dropout(0.2)
    )

    # in: (batch size, 1, length, d_model)
    # out: (batch size, 1, length, d_model)
    self.convolution = nn.Sequential(
        nn.LayerNorm(d_model),
        # Point-wise conv
        nn.Conv2d(1, 12, 1, 1, 0),
        nn.GLU(dim=1),
        # Depth-wise conv
        nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1, stride=1, groups=6),# group means separate kernels for each channel.
        nn.BatchNorm2d(6),
        nn.Hardswish(),
        # Point-wise conv
        nn.Conv2d(6, 1, 1, 1, 0),
        nn.Dropout(0.1)
    )

    self.layerNorm = nn.LayerNorm(d_model)

  def forward(self, inputs):
    """
    args:
      inputs: (batch size, length, d_model)
    return:
      out: (batch size, n_spks)
    """
    # out: (batch size, length, )
    out = inputs + 0.5 * self.feedForward1(inputs)
    # out: (length, batch size, d_model)
    out = out.permute(1, 0, 2)
    # out: (length, batch size, d_model)
    out = out + self.multiHeaded(out)
    # out: (batch size, length, d_model)
    out = out.permute(1, 0, 2)
    out = out.unsqueeze(1)
    # out: (batch size, 1, length, d_model)
    out = out + self.convolution(out)
    # out: (batch size, length, d_model)
    out = out.squeeze(1)
    # out: (batch size, length, d_model)
    out = out + 0.5 * self.feedForward2(out)
    out = self.layerNorm(out)
    return out

class Classifier(nn.Module):
  def __init__(self, d_model=256, n_spks=600, dropout=0.1):
    super().__init__()
    # Project the dimension of features from that of input into d_model.
    self.prenet = nn.Linear(40, d_model)
    # TODO:
    #   Change Transformer to Conformer.
    #   https://arxiv.org/abs/2005.08100
    self.encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model, dim_feedforward=256, nhead=2
    )
    # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

    self.conformer1 = Conformer()
    self.conformer2 = Conformer()
    # self.conformer3 = Conformer()
    # self.conformer4 = Conformer()

    
    # Project the the dimension of features from d_model into speaker nums.
    self.pred_layer = nn.Sequential(
      nn.Linear(d_model, d_model),
      nn.LeakyReLU(),
      nn.Linear(d_model, n_spks),
    )

    # strange attention pooling https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1158.pdf
    # out: (batch size, length, d_model)
    self.attnpool = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.LeakyReLU(),
        nn.Linear(d_model, 1),
        nn.Softmax(dim=1),
    )

  def forward(self, mels):
    """
    args:
      mels: (batch size, length, 40)
    return:
      out: (batch size, n_spks)
    """
    # out: (batch size, length, d_model)
    out = self.prenet(mels)
    # out: (batch size, length, d_model)
    out = self.conformer1(out)
    # out: (batch size, length, d_model)
    out = self.conformer2(out)
    # # out: (batch size, length, d_model)
    # out = self.conformer3(out)
    # # out: (batch size, length, d_model)
    # out = self.conformer4(out)
    
    # # mean pooling
    # stats = out.mean(dim=1)
    # strange attention pooling https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1158.pdf
    # out: (batch size, length, 1)
    stats = self.attnpool(out)

    # out: (batch size, d_model)
    out = torch.bmm(out.permute(0, 2, 1), stats)
    out = torch.squeeze(out, 2)

    # out: (batch, n_spks)
    out = self.pred_layer(out)

    return out
  # def forward(self, mels):
  #   """
  #   args:
  #     mels: (batch size, length, 40)
  #   return:
  #     out: (batch size, n_spks)
  #   """
  #   # out: (batch size, length, d_model)
  #   out = self.prenet(mels)
  #   # out: (length, batch size, d_model)
  #   out = out.permute(1, 0, 2)
  #   # The encoder layer expect features in the shape of (length, batch size, d_model).
  #   out = self.encoder_layer(out)
  #   # out: (batch size, length, d_model)
  #   out = out.transpose(0, 1)
  #   # mean pooling
  #   stats = out.mean(dim=1)

  #   # out: (batch, n_spks)
  #   out = self.pred_layer(stats)
  #   return out

"""# Learning rate schedule
- For transformer architecture, the design of learning rate schedule is different from that of CNN.
- Previous works show that the warmup of learning rate is useful for training models with transformer architectures.
- The warmup schedule
  - Set learning rate to 0 in the beginning.
  - The learning rate increases linearly from 0 to initial learning rate during warmup period.
"""

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
  optimizer: Optimizer,
  num_warmup_steps: int,
  num_training_steps: int,
  num_cycles: float = 0.5,
  last_epoch: int = -1,
):
  """
  Create a schedule with a learning rate that decreases following the values of the cosine function between the
  initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
  initial lr set in the optimizer.

  Args:
    optimizer (:class:`~torch.optim.Optimizer`):
      The optimizer for which to schedule the learning rate.
    num_warmup_steps (:obj:`int`):
      The number of steps for the warmup phase.
    num_training_steps (:obj:`int`):
      The total number of training steps.
    num_cycles (:obj:`float`, `optional`, defaults to 0.5):
      The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
      following a half-cosine).
    last_epoch (:obj:`int`, `optional`, defaults to -1):
      The index of the last epoch when resuming training.

  Return:
    :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
  """

  def lr_lambda(current_step):
    # Warmup
    if current_step < num_warmup_steps:
      return float(current_step) / float(max(1, num_warmup_steps))
    # decadence
    progress = float(current_step - num_warmup_steps) / float(
      max(1, num_training_steps - num_warmup_steps)
    )
    return max(
      0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    )

  return LambdaLR(optimizer, lr_lambda, last_epoch)

"""# Model Function
- Model forward function.
"""

import torch


def model_fn(batch, model, criterion, device):
  """Forward a batch through the model."""

  mels, labels = batch
  mels = mels.to(device)
  labels = labels.to(device)

  outs = model(mels)

  loss = criterion(outs, labels)

  # Get the speaker id with highest probability.
  preds = outs.argmax(1)
  # Compute accuracy.
  accuracy = torch.mean((preds == labels).float())

  return loss, accuracy

"""# Validate
- Calculate accuracy of the validation set.
"""

from tqdm import tqdm
import torch


def valid(dataloader, model, criterion, device): 
  """Validate on validation set."""

  model.eval()
  running_loss = 0.0
  running_accuracy = 0.0
  pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

  for i, batch in enumerate(dataloader):
    with torch.no_grad():
      loss, accuracy = model_fn(batch, model, criterion, device)
      running_loss += loss.item()
      running_accuracy += accuracy.item()

    pbar.update(dataloader.batch_size)
    pbar.set_postfix(
      loss=f"{running_loss / (i+1):.2f}",
      accuracy=f"{running_accuracy / (i+1):.2f}",
    )

  pbar.close()
  model.train()

  return running_accuracy / len(dataloader)

"""# Main function"""

from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split


def parse_args():
  """arguments"""
  config = {
    "data_dir": "./Dataset",
    "save_path": "model.ckpt",
    "batch_size": 32,
    "n_workers": 8,
    "valid_steps": 2000,
    "warmup_steps": 1000,
    "save_steps": 10000,
    "total_steps": 100000,
  }

  return config


def main(
  data_dir,
  save_path,
  batch_size,
  n_workers,
  valid_steps,
  warmup_steps,
  total_steps,
  save_steps,
):
  """Main function."""
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"[Info]: Use {device} now!")

  train_loader, valid_loader, speaker_num = get_dataloader(data_dir, batch_size, n_workers)
  train_iterator = iter(train_loader)
  print(f"[Info]: Finish loading data!",flush = True)

  model = Classifier(n_spks=speaker_num).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = AdamW(model.parameters(), lr=1e-3)
  scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
  print(f"[Info]: Finish creating model!",flush = True)

  best_accuracy = -1.0
  best_state_dict = None

  pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

  for step in range(total_steps):
    # Get data
    try:
      batch = next(train_iterator)
    except StopIteration:
      train_iterator = iter(train_loader)
      batch = next(train_iterator)

    loss, accuracy = model_fn(batch, model, criterion, device)
    batch_loss = loss.item()
    batch_accuracy = accuracy.item()

    # Updata model
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
    # Log
    pbar.update()
    pbar.set_postfix(
      loss=f"{batch_loss:.2f}",
      accuracy=f"{batch_accuracy:.2f}",
      step=step + 1,
    )

    # Do validation
    if (step + 1) % valid_steps == 0:
      pbar.close()

      valid_accuracy = valid(valid_loader, model, criterion, device)

      # keep the best model
      if valid_accuracy > best_accuracy:
        best_accuracy = valid_accuracy
        best_state_dict = model.state_dict()

      pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    # Save the best model so far.
    if (step + 1) % save_steps == 0 and best_state_dict is not None:
      torch.save(best_state_dict, save_path)
      pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

  pbar.close()


if __name__ == "__main__":
  main(**parse_args())

"""# Inference"""



"""## Dataset of inference"""

import os
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset


class InferenceDataset(Dataset):
  def __init__(self, data_dir):
    testdata_path = Path(data_dir) / "testdata.json"
    metadata = json.load(testdata_path.open())
    self.data_dir = data_dir
    self.data = metadata["utterances"]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    utterance = self.data[index]
    feat_path = utterance["feature_path"]
    mel = torch.load(os.path.join(self.data_dir, feat_path))

    return feat_path, mel


def inference_collate_batch(batch):
  """Collate a batch of data."""
  feat_paths, mels = zip(*batch)

  return feat_paths, torch.stack(mels)

"""## Main funcrion of Inference"""

import json
import csv
from pathlib import Path
from tqdm.notebook import tqdm

import torch
from torch.utils.data import DataLoader

def parse_args():
  """arguments"""
  config = {
    "data_dir": "./Dataset",
    "model_path": "./model.ckpt",
    "output_path": "./output.csv",
  }

  return config


def main(
  data_dir,
  model_path,
  output_path,
):
  """Main function."""
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"[Info]: Use {device} now!")

  mapping_path = Path(data_dir) / "mapping.json"
  mapping = json.load(mapping_path.open())

  dataset = InferenceDataset(data_dir)
  dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=8,
    collate_fn=inference_collate_batch,
  )
  print(f"[Info]: Finish loading data!",flush = True)

  speaker_num = len(mapping["id2speaker"])
  model = Classifier(n_spks=speaker_num).to(device)
  model.load_state_dict(torch.load(model_path))
  model.eval()
  print(f"[Info]: Finish creating model!",flush = True)

  results = [["Id", "Category"]]
  for feat_paths, mels in tqdm(dataloader):
    with torch.no_grad():
      mels = mels.to(device)
      outs = model(mels)
      preds = outs.argmax(1).cpu().numpy()
      for feat_path, pred in zip(feat_paths, preds):
        results.append([feat_path, mapping["id2speaker"][str(pred)]])
  
  with open(output_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(results)


if __name__ == "__main__":
  main(**parse_args())

