import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch
from LibriDatasetEng import LibriDatasetEng
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import torch
import torchaudio
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()
alphabet =  ' abcdefghijklmnopqrstuvwxyz\'-'

print(len(alphabet))
train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35)
)

class LinearModule(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModule, self).__init__()
        self.lin = nn.Linear(input_size,output_size)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.lin(x)
        x = self.relu(x)
        return x
    
class ConvModule(nn.Module):
    def __init__(self, in_c, o_ch, kernel_size, dropout):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_c, o_ch, kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x
    
import torch
import torch.nn as nn
from torch.nn import functional as F



class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.lin0 = LinearModule(128, 128)
        self.conv1 = nn.Conv2d(1,32, 3)
        self.conv2 = ConvModule(32,32,3,0.1)
        self.conv3 = ConvModule(32,32,3,0.1)
        self.conv4 = ConvModule(32,32,3,0.1)
        self.lin1 = LinearModule(120, 128)
        self.lin2 = LinearModule(128, 128)
        self.lstm = nn.LSTM(128, 10, 10)
        self.lin3 = LinearModule(128, 30)
        self.logsoftmax = nn.LogSoftmax(2)
    def forward(self, x):
        x = self.lin0(x)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = self.lin1(x)
        # x = x.unsqueeze(1)
        # h0 = torch.zeros(10, 63744, 10).to(device)
        # c0 = torch.zeros(10, 63744, 10).to(device)
        # x, a= self.lstm(x, (h0,c0))
        x = self.lin3(x)
        x = self.logsoftmax(x)
        return x
    
class SpeechRecognizer:
    def __init__(self):
        wds_train = LibriDatasetEng()
        self.wds_dl = DataLoader(wds_train, batch_size=1, shuffle=True)
        self.model = Network().to(device)

        # summary(self.model, (3000,128))
        self.loss_fn = torch.nn.CTCLoss(blank=29)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
    def train_one_epoch(self, epoch_index):
     epoch_losses = []
     for i, data in enumerate(self.wds_dl):
        inputs, labels, input_lengths, labels_lengths = data
        input_lengths = torch.tensor(input_lengths)
        labels_lengths = torch.tensor(labels_lengths)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        outputs = outputs.transpose(0,1)
        loss = self.loss_fn(outputs.float(), labels.float(), input_lengths, labels_lengths)
        loss.backward()
        self.optimizer.step()
        epoch_losses.append(loss.item())
        print(loss.item())
     return np.mean(epoch_losses)
    
    def train(self, epochs):
        for i in range(epochs):
            print(f'{i} epoch')
            loss = self.train_one_epoch(i)
            print(loss)

    def predict(self, wav_path):
        y, sr = torchaudio.load(wav_path)
        y = train_audio_transforms(y)
        
        y = torch.tensor(np.pad(y,((0,0),(0, 0), (0,1500-y.shape[2])), mode='constant',constant_values=((0,0),(0,0), (0,0)))).squeeze(0)
        y = y.transpose(0,1).to(device)
        out = y.view(1, y.shape[0], y.shape[1])
        out = self.model(out)
        out = out.squeeze()
        result=''
        for i in out:
            if torch.argmax(i) == 30:
                result+='%'
            else:
                result+=alphabet[torch.argmax(i)]
        print(result.replace('%', '').replace(' ', ''))
            






gc = SpeechRecognizer()
gc.train(10)
gc.predict('dataset\\train\\audio\\295\\162\Leo-Tolstoy-Detstvo-RUSSIAN-28-Posledniye_Grustnye_Vospominaniya_0140.wav')

