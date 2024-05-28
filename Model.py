import torch, torchaudio
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from DownloadDataset import LibriDataset
import pickle
from EvaluationFuncs import cer, wer


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def int_to_text(tsequence):
    alphabet =  ' абвгдеёжзийклмнопрстуфхцчшщьыъэюя'
    result = ''
    for i in tsequence:
        result+=alphabet[i]
    return result

def GreedyDecoder(output, labels, label_lengths, blank_label=34, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        targets.append(int_to_text(labels[i].int().tolist()[:label_lengths[i]]))
        decode = []
        for j, index in enumerate(args):
            if index!=blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(int_to_text(decode))
    return decodes, targets

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

class LSTM(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.lstm(x)
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
        self.lstm_layers = nn.Sequential(*[
            LSTM(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
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
        x = self.lstm_layers(x)
        x = self.classifier(x)
        return x
    

class SpeechRecognizer:
    def __init__(self):
        # wds_train = LibriDataset(30000, 'train')
        # self.wds_dl = DataLoader(dataset=wds_train, batch_size=10, shuffle=True)
        # self.test_dataloader = DataLoader(dataset = LibriDataset('test'), batch_size=10, shuffle=False)
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
        wds_train = LibriDataset(30000, 'train')
        self.wds_dl = DataLoader(dataset=wds_train, batch_size=10, shuffle=True)
        for i in range(epochs):
            print(f'{i} epoch')
            loss = self.train_one_epoch(i)
            print(loss)

    def test(self):
        self.test_dataloader = DataLoader(dataset = LibriDataset(300, 'test'), batch_size=10, shuffle=False)
        test_loss = 0
        test_cer, test_wer = [], []
        with torch.no_grad():
            for i, data in enumerate(self.test_dataloader):
                specs, labels, input_lengths, label_lengths = data
                outputs = self.model(specs)
                outputs = F.log_softmax(outputs, dim=2)
                outputs = outputs.transpose(0, 1) # (time, batch, n_class)
                loss = self.loss_fn(outputs, labels, input_lengths, label_lengths)
                test_loss += loss.item() / len(self.test_dataloader)
                preds, targets = GreedyDecoder(outputs.transpose(0,1), labels, label_lengths)
                print(preds, targets)
                for j in range(len(preds)):
                    test_cer.append(cer(targets[j], preds[j]))
                    test_wer.append(wer(targets[j], preds[j]))
        avg_cer = np.mean(test_cer)
        avg_wer = np.mean(test_wer)
        print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))
    def save_model(self):
        torch.save(self.model, 'trained_model.pt')
    def load_model(self, path='trained_model.pt'):
        self.model = torch.load('trained_model.pt')

# sr = SpeechRecognizer()
# sr.train(100)
# sr.save_model()
# sr.test()