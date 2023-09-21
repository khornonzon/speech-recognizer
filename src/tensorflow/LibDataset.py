import librosa
import numpy as np
import tensorflow
from tqdm import tqdm
import os
class LibriDatasetEng():
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
                        y, sr = librosa.load(path + file)
                        y = librosa.feature.melspectrogram(y=y, sr=sr)
                        self.input_lengths.append(y.shape[1])
                        y = np.pad(y,((0,0),(0,3000-y.shape[1])), mode='constant',constant_values=((0,0),(0,0)))
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
                    self.int_targets.append(int_target)
                    self.label_lengths.append(len(int_target))
        self.labels = tensorflow.Tensor(self.int_targets)
        self.specs = tensorflow.Tensor(self.specs)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        return  self.specs[index], self.labels[index], self.input_lengths[index], self.label_lengths[index]
