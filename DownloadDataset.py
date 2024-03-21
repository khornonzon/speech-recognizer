from torch.utils.data import Dataset, DataLoader
import torchaudio
import json
import io
import codecs
import torch
import torch.nn as nn
import librosa
import numpy as np
from tqdm import tqdm
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100)
)
test_audio_transforms = torchaudio.transforms.MelSpectrogram()

class LibriDataset(Dataset):
    def __init__(self, size, tt_mode='train'):
        path_json = tt_mode + '\\manifest.json'
        file = open(path_json, 'r', encoding='utf-8')
        self.file_names = []
        self.text_targets = []
        self.int_targets= []
        self.input_lengths = []
        self.label_lengths = []
        self.alphabet =  ' абвгдеёжзийклмнопрстуфхцчшщьыъэюя'
        self.slice = size
        for s in file:
            json_object = json.loads(s)
            self.file_names.append(tt_mode + '/'+json_object['audio_filepath'])
            self.text_targets.append(json_object['text'])

        c = list(zip(self.file_names, self.text_targets))
        random.shuffle(c)
        self.file_names, self.text_targets = zip(*c)
        for target in self.text_targets:
            int_target = [self.alphabet.index(c) for c in target.lower()]
            self.int_targets.append(torch.tensor(int_target))
            self.label_lengths.append(len(int_target))
        self.labels = nn.utils.rnn.pad_sequence(self.int_targets, batch_first=True)[:self.slice].float()
        self.specs =[] 
        for i in tqdm(range(self.slice)):
            y, sr = torchaudio.load(self.file_names[i])
            if tt_mode == 'train':
                y = train_audio_transforms(y).squeeze(0).transpose(0, 1)
            else:
                y = test_audio_transforms(y).squeeze(0).transpose(0,1)
            self.specs.append(y)
            self.input_lengths.append(y.shape[0]//2)
        self.specs = nn.utils.rnn.pad_sequence(self.specs, batch_first=True).unsqueeze(1).transpose(2, 3)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        return  self.specs[index].to(device), self.labels[index].to(device), self.input_lengths[index], self.label_lengths[index]
