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
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35)
)
class LibriDatasetEng(Dataset):
    def __init__(self, path='dataset\eng\LibriSpeech\dev-clean'):
        self.file_names = []
        self.text_targets = []
        self.int_targets= []
        self.input_lengths = []
        self.label_lengths = []
        self.specs = []
        self.alphabet =  ' abcdefghijklmnopqrstuvwxyz\'-'
        directories = os.listdir(path)
        for direct in tqdm(directories):
            path='dataset\eng\LibriSpeech\dev-clean'
            subdirs = os.listdir(path + '\\' + direct)
            for subdir in subdirs:
                path='dataset\eng\LibriSpeech\dev-clean'
                path = path + '\\' + direct + '\\' + subdir + '\\'
                txt_file = ''
                for file in os.listdir(path):
                    if file.endswith('.txt'):
                        txt_file = path+file
                    else:
                        y, sr = torchaudio.load(path + file)
                        y = train_audio_transforms(y)
                        self.input_lengths.append(y.shape[2])
                        y = torch.tensor(np.pad(y,((0,0),(0, 0), (0,3000-y.shape[2])), mode='constant',constant_values=((0,0),(0,0), (0,0)))).squeeze(0)
                        self.specs.append(y.transpose(0,1))

                txt_file = open(txt_file, 'r')
                for line in txt_file:
                    text = line.split(' ')[1:len(line.split(' '))]
                    string = ""
                    for word in text:
                        string=string + word + ' '
                    string = string[0:len(string)-1]
                    string = string.replace('\n', '')
                    int_target = [self.alphabet.index(c) for c in string.lower()]
                    self.int_targets.append(torch.tensor(int_target))
                    self.label_lengths.append(len(int_target))
        self.labels = nn.utils.rnn.pad_sequence(self.int_targets, batch_first=True)
        self.specs = self.specs
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        return  self.specs[index].to(device), self.labels[index].to(device), self.input_lengths[index], self.label_lengths[index]
