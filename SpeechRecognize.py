import torch, torchaudio
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from DownloadDataset import LibriDataset
import pickle
from EvaluationFuncs import cer, wer
from Model import *
test_audio_transforms = torchaudio.transforms.MelSpectrogram()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def int_to_text(tsequence):
    alphabet =  ' абвгдеёжзийклмнопрстуфхцчшщьыъэюя'
    result = ''
    for i in tsequence:
        result+=alphabet[i]
    return result

def decoder(output, blank_label=34, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    for i, args in enumerate(arg_maxes):
        decode = []
        for j, index in enumerate(args):
            if index!=blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(int_to_text(decode))
    return decodes



class SpeechRecognize:
    def __init__(self):
        self.model = torch.load('trained_model.pt')
    
    def recognize(self, file_name):
        y, sr = torchaudio.load(file_name)
        y = test_audio_transforms(y).squeeze(0).transpose(0,1)
        y = y.unsqueeze(0)
        y = y.unsqueeze(1).transpose(2, 3).to(device)

        output = self.model(y)

        recognized_text = decoder(output)[0]
        return recognized_text
    
sr = SpeechRecognize()
print(sr.recognize('train/audio/295/162/Leo-Tolstoy-Detstvo-RUSSIAN-01-Karl-Ivanych_0001.wav'))
        