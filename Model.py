import torch, torchaudio
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from DownloadDataset import LibriDataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CNNLayerNorm(nn.Module):
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 

class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)
        
class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x



class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x

class LinearModule(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModule, self).__init__()
        self.lin = nn.Linear(input_size,output_size)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.lin(x)
        x = self.relu(x)
        return x
    

class SpeechRecognizer:
    def __init__(self):
        wds_train = LibriDataset()
        self.wds_dl = DataLoader(dataset=wds_train, batch_size=10, shuffle=True)
        self.model =  SpeechRecognitionModel(3,5,512,35,128).to(device)
        # summary(self.model, (3000,128))
        self.loss_fn = torch.nn.CTCLoss(blank=34).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4)

    def train_one_epoch(self, epoch_index):
        self.model.train()
        data_len = len(self.wds_dl.dataset)
        epoch_losses = []
        for batch_idx, data in enumerate(self.wds_dl):
            inputs, labels, input_lengths, labels_lengths = data
            input_lengths = torch.tensor(input_lengths)
            labels_lengths = torch.tensor(labels_lengths)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            outputs = F.log_softmax(outputs, dim=2)
            outputs = outputs.transpose(0,1)
            loss = self.loss_fn(outputs, labels, input_lengths, labels_lengths)
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()
            epoch_losses.append(loss.item())
            if batch_idx % 100 == 0 or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch_index, batch_idx * len(inputs), data_len,
                    100. * batch_idx / len(self.wds_dl), loss.item()))
        return np.mean(epoch_losses)
    
    def train(self, epochs):
        for i in range(epochs):
            print(f'{i} epoch')
            loss = self.train_one_epoch(i)
            print(loss)
sr = SpeechRecognizer()
sr.train(10)