import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from src.dataset import LJDataset
from src.wavenet import WaveNet, encode_mu_law, quantize
from config import ModelConfig
from src.preprocessing import MelSpectrogram
from config import MelSpectrogramConfig
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("LJSpeech-1.1/metadata.csv", sep='|', quotechar='`', index_col=0, header=None)
train, test = train_test_split(df, test_size=0.2, random_state=10)

train_dataset = LJDataset(train)
test_dataset = LJDataset(test)

model_config = ModelConfig()

train_dataloader = DataLoader(train_dataset,
                              batch_size=model_config.batch_size,
                              num_workers=model_config.num_workers,
                              shuffle=False,
                              pin_memory=True)
test_dataloader = DataLoader(test_dataset,
                             batch_size=model_config.batch_size,
                             num_workers=model_config.num_workers,
                             pin_memory=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = WaveNet()

if model_config.from_pretrained:
    model.load_state_dict(torch.load(model_config.model_path))
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.lr)
num_steps = len(train_dataloader) * model_config.num_epochs
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=0.00001)

if model_config.wandb_log:
    wandb.init(project="wavenet")
    wandb.watch(model, log="all")

featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)
best_loss = 10.0

for epoch in range(1, model_config.num_epochs + 1):
    model.to(device)
    train_losses = []
    train_accuracy = []
    model.train()
    for wavs in train_dataloader:
        wavs = wavs.to(device)

        zero_frame = torch.zeros((wavs.shape[0], 1)).to(device)
        padded_wavs = torch.cat([zero_frame, wavs[:, :-1]], dim=1)

        mels = featurizer(wavs)
        outputs = model(padded_wavs, mels)

        classes = outputs.argmax(dim=1)
        quantized_wavs = quantize(encode_mu_law(wavs))
        loss = nn.CrossEntropyLoss()(outputs, quantized_wavs)

        accuracy = (classes == quantized_wavs).sum().item() / classes.shape[-1] / classes.shape[0]
        train_accuracy.append(accuracy)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        lr_scheduler.step()
        train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_losses = []
        val_accuracy = []
        for wavs in test_dataloader:
            wavs = wavs.to(device)
            zeros = torch.zeros((wavs.shape[0], 1)).to(device)
            padded_wavs = torch.cat([zeros, wavs[:, :-1]], dim=1)

            mels = featurizer(wavs)

            outputs = model(padded_wavs, mels)
            classes = outputs.argmax(dim=1)
            quantized_wavs = quantize(encode_mu_law(wavs))

            accuracy = (classes == quantized_wavs).sum().item() / classes.shape[-1] / classes.shape[0]
            val_accuracy.append(accuracy)

            loss = nn.CrossEntropyLoss()(outputs, quantized_wavs)
            val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    train_acc = np.mean(train_accuracy)
    val_acc = np.mean(val_accuracy)

    if model_config.wandb_log:
        wandb.log({"train_loss": train_loss,
                   "val_loss": val_loss,
                   "train accuracy": train_acc,
                   "val accuracy": val_acc,
                   })

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "wavenet.pth")
