import os
import json
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class myDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, segment_len=128):
        self.data_dir = data_dir
        self.segment_len = segment_len
        mapping_path = Path(data_dir) / 'mapping.json'
        mapping = json.load(mapping_path.open())
        self.speaker2id = mapping['speaker2id']
        metadata_path = Path(data_dir) / 'metadata.json'
        metadata = json.load(open(metadata_path))['speakers']
        self.speaker_num = len(metadata.keys())
        self.data = []
        for speaker in metadata.keys():
            for utterances in metadata.keys():
                self.data.append([utterances['feature_path'], self.speaker2id[speaker]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feat_path, speaker = self.data[index]
        mel = torch.load(os.path.join(self.data_dir, feat_path))
        if len(mel) > self.segment_len:
            start = random.randint(0, len(mel) - self.segment_len)
            mel = torch.FloatTensor(mel[start:start + self.segment_len])
        else:
            mel = torch.FloatTensor(mel)
        speaker = torch.FloatTensor([speaker]).long
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
    mel, speaker = zip(*batch)
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)
    return mel, torch.FloatTensor(speaker).long()


def get_data_loader(data_dir, batch_size, n_workers):
    dataset = myDataset(data_dir)
    speaker_num=dataset.get_speaker_number()
    trainlen=int(0.9*len(dataset))
    lengths=[trainlen,len(dataset)-trainlen]
    trainset,validset=random_split(dataset,lengths)
    train_loader=DataLoader(
        trainset,
        batch_size=batch_size,
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


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        testdata_path = Path(data_dir) / 'testdata.json'
        metadata = json.load(testdata_path.open())
        self.data_dir = data_dir
        self.data = metadata['utterances']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utterance = self.data[index]
        feat_path = utterance['feature_path']
        mel = torch.load(os.path.join(self.data_dir, feat_path))
        return feat_path, mel


class Classifier(torch.nn.Module):
    def __init__(self, d_model=80, n_spks=600, dropout=0.1):
        super.__init__()
        self.prenet = torch.nn.Linear(40, d_model)
        self.encoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=2)
        self.pred_layer = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, n_spks),
        )
    def forward(self,mels):
        out=self.prenet(mels)
        out=out.permute(1,0,2)
        out=self.encoder_layer(out)
        out=out.transpose(0,1)
        stats=out.mean(dim=1)
        out=self.pred_layer(stats)
        return out


def inference_collate_batch(batch):
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
    config = {
        "data_dir": "../data/Dataset",
        "model_path": "./model.ckpt",
        "output_path": "./output.csv",
    }
    return config


def main(
        data_dir,
        model_path,
        output_path,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[info]: use {device} now!")

    mapping_path = Path(data_dir) / 'mapping.json'
    mapping = json.load(mapping_path.open())

    print(mapping_path)
    print(mapping)
    dataset = InferenceDataset(data_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        collate_fn=inference_collate_batch,
    )
    print(f"[Info] : Finsih loading data!", flush=True)
    speacker_num = len(mapping['id2speaker'])
    model = Classifier(n_spks=speacker_num).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"[info]:Finish creating model!", flush=True)
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


if __name__ == '__main__':
    main(**parse_args())
