import os
from torch.utils.data import Dataset
import torchaudio
from src.preprocessing import MelSpectrogram
from config import MelSpectrogramConfig
import numpy as np

vocab = " abcdefghijklmnopqrstuvwxyz.,:;-?!"

charToIdx = {c: i for i, c in enumerate(vocab)}
idxToChar = {i: c for i, c in enumerate(vocab)}


class LJDataset(Dataset):
    def __init__(self, df):
        self.dir = "LJSpeech-1.1/wavs"
        self.filenames = df.index.values
        self.labels = df[2].values
        self.featurizer = MelSpectrogram(MelSpectrogramConfig())

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        wav, sr = torchaudio.load(os.path.join(self.dir, f'{filename}.wav'))
        wav = wav.squeeze()
        position = np.random.choice(wav.shape[0] - 20000)
        return wav[position:position + 20000]

    def __len__(self):
        return len(self.filenames)
