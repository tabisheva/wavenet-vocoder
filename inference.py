import torch
import numpy as np
import wandb
from src.wavenet import WaveNet
from config import ModelConfig, MelSpectrogramConfig
from src.preprocessing import MelSpectrogram
import torchaudio
import warnings
warnings.filterwarnings('ignore')
import sys


model_config = ModelConfig()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model_config = ModelConfig()
model = WaveNet()
model.load_state_dict(torch.load(model_config.model_path, map_location=device))
model.to(device)

wav, sr = torchaudio.load("LJ045-0097.wav")
wav = wav.to(device)
featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)

model.eval()
with torch.no_grad():

    mel = featurizer(wav)
    y_pred = model.inference(mel).squeeze()

if model_config.wandb_log:
    wandb.init(project="wavenet")
    wandb.log({
               "inference audio": [wandb.Audio(y_pred.cpu(), sample_rate=22050)],
               "GT audio": [wandb.Audio(wav.squeeze().cpu(), sample_rate=22050)],
               })
