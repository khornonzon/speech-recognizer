import sys
import pyaudio
import wave
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel
from PyQt5.QtCore import QThread, pyqtSignal
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
    

class SpeechRecognitionThread(QThread):
    recognized_text = pyqtSignal(str)
    
    def __init__(self, model, filename):
        super().__init__()
        self.model = model
        self.filename = filename

    def run(self):
        text = self.model.recognize(self.filename)  # Вызов метода распознавания модели с wav-файлом
        self.recognized_text.emit(text)

class SpeechRecognitionApp(QWidget):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.is_recording = False
        self.initUI()
        self.stream = None
        self.frames = []

    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel("Распознанный текст:")
        layout.addWidget(self.label)

        self.textEdit = QTextEdit()
        layout.addWidget(self.textEdit)

        self.startButton = QPushButton('Начать запись')
        self.startButton.clicked.connect(self.start_recording)
        layout.addWidget(self.startButton)

        self.stopButton = QPushButton('Остановить запись и распознать')
        self.stopButton.clicked.connect(self.stop_recording)
        layout.addWidget(self.stopButton)

        self.setLayout(layout)
        self.setWindowTitle('Распознавание речи')
        self.resize(400, 300)

    def start_recording(self):
        self.is_recording = True
        self.frames = []

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=44100,
                                  input=True,
                                  frames_per_buffer=1024)

        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)

        self.record_audio()

    def record_audio(self):
        while self.is_recording:
            data = self.stream.read(1024)
            self.frames.append(data)
            QApplication.processEvents()

    def stop_recording(self):
        self.is_recording = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)

        # Сохранение записи в wav-файл
        filename = 'output.wav'
        wf = wave.open(filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        # Запуск распознавания
        self.speech_thread = SpeechRecognitionThread(self.model, filename)
        self.speech_thread.recognized_text.connect(self.update_text)
        self.speech_thread.start()

    def update_text(self, text):
        self.textEdit.append(text)

if __name__ == '__main__':
    class MockModel:
        def recognize(self, filename):
            # Заглушка для демонстрации
            return f"Распознанный текст из {filename}"

    app = QApplication(sys.argv)
    model = SpeechRecognize()
    ex = SpeechRecognitionApp(model)
    ex.show()
    sys.exit(app.exec_())
