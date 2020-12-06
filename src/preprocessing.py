import torch
import torch.nn as nn
import torchaudio
import librosa
from config import MelSpectrogramConfig


class MelSpectrogram(nn.Module):

    def __init__(self, config: MelSpectrogramConfig):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels
        )

        self.mel_spectrogram.spectrogram.power = config.power

        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """
        mel = self.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

        return mel


def collate_fn(batch):
    """
    Stacking sequences of variable lengths in batches
    :param batch: list of tuples with texts (num_chars, ) and mels (num_channels, T)
    :return: tensor (batch, max_length of texts) with zero-padded texts,
             tensor (batch, ) with texts lengths,
             tensor (batch, num_channels, max_time) with padded mels,
             tensor (batch, max_time) with padded gates
             tensor (batch, ) with mel lengths
    """
    wavs = [x[:20000] for x in batch]
    # (B, 20000)
    wavs_padded = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True).long()

    return wavs_padded


def get_mask_from_lengths(lengths, device):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len)).to(device)
    mask = (ids < lengths.unsqueeze(1))
    return mask
