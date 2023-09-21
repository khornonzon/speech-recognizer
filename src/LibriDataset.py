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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)
#     torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
#     torchaudio.transforms.TimeMasking(time_mask_param=35)
)
class LibriDataset(Dataset):
    def __init__(self, path_json='dataset\\train\manifest.json'):
        file = open(path_json, 'r', encoding='utf-8')
        self.file_names = []
        self.text_targets = []
        self.int_targets= []
        self.input_lengths = []
        self.label_lengths = []
        self.alphabet =  ' абвгдеёжзийклмнопрстуфхцчшщьыъэюя'
        self.slice = 1000
        print(len(self.alphabet))
        for s in file:
            json_object = json.loads(s)
            self.file_names.append('dataset/train/'+json_object['audio_filepath'])
            self.text_targets.append(json_object['text'])
        for target in self.text_targets:
            int_target = [self.alphabet.index(c) for c in target.lower()]
            self.int_targets.append(torch.tensor(int_target))
            self.label_lengths.append(len(target))
        self.labels = nn.utils.rnn.pad_sequence(self.int_targets, batch_first=True)[:self.slice]
        self.specs =[] 
        for i in tqdm(range(self.slice)):
            y, sr = torchaudio.load(self.file_names[i])
            y = train_audio_transforms(y)
            self.input_lengths.append(y.shape[2])
            y = torch.tensor(np.pad(y,((0,0),(0, 0), (0,2000-y.shape[2])), mode='constant',constant_values=((0,0),(0,0), (0,0)))).squeeze(0)
            self.specs.append(y.transpose(0,1))




    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        return  self.specs[index].to(device), self.labels[index].to(device), self.input_lengths[index], self.label_lengths[index]
    