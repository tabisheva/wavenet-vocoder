import torch
import torch.nn as nn
import numpy as np


def encode_mu_law(x):
    # Encode audio to [-1, 1] with 256 values
    mu_law = torch.sign(x) * torch.log(1 + 255 * torch.abs(x)) / np.log(256.)
    return mu_law


def decode_mu_law(mu_law):
    # Decode mu_law
    decoded_x = torch.sign(mu_law) / 255 * (torch.pow(256, torch.abs(mu_law)) - 1)
    return decoded_x


def quantize(mu_law):
    # Convert from [-1, 1] to [0, 255]
    quantized_x = torch.floor((mu_law + 1) / 2 * 255 + 0.5).to(torch.long)
    return quantized_x


def dequantize(quantized_x):
    # Convert from [0, 256] to [-1, 1]
    dequantized_x = (quantized_x - 0.5) / 255 * 2 - 1
    return dequantized_x


class CausalConv1d(nn.Module):
    # Dilated convolutions with shifting
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        if self.padding != 0:
            x = x[:, :, :-self.padding]
        # (B, C, T)
        return x


class WaveNetBlock(nn.Module):

    def __init__(self, dilation):
        super(WaveNetBlock, self).__init__()

        self.dil_sigmoid = CausalConv1d(120, 120, kernel_size=2, dilation=dilation)
        self.dil_tanh = CausalConv1d(120, 120, kernel_size=2, dilation=dilation)
        self.mel_1x1_sigmoid = nn.Conv1d(80, 120, kernel_size=1)
        self.mel_1x1_tanh = nn.Conv1d(80, 120, kernel_size=1)
        self.skip_1x1 = nn.Conv1d(120, 240, kernel_size=1)
        self.res_1x1 = nn.Conv1d(120, 120, kernel_size=1)

    def forward(self, x, h):
        """
        :param x: audio signal after first convolution with shape (B, 120, T)
        :param h: upsampled mel spec with shape (B, num_mels, T)
        """
        output_sigmoid = self.dil_sigmoid(x)
        output_tanh = self.dil_tanh(x)
        mel_output_sigmoid = self.mel_1x1_sigmoid(h)
        mel_output_tanh = self.mel_1x1_tanh(h)
        output = torch.sigmoid(output_sigmoid + mel_output_sigmoid) * torch.tanh(output_tanh + mel_output_tanh)
        skip = self.skip_1x1(output)  # (B, 240, T)
        output = self.res_1x1(output)  # (B, 120, T)
        output += x
        return output, skip


class WaveNet(nn.Module):

    def __init__(self, num_mels=80, kernel_size=2, residual_channels=120, skip_channels=240,
                 dilation_depth=8, dilation_repeat=2, quantized_values=256):
        super(WaveNet, self).__init__()
        self.dilations = [2 ** i for i in range(dilation_depth)] * dilation_repeat

        self.receptive_field = (kernel_size - 1) * sum(self.dilations) + 1

        self.start_conv = nn.Conv1d(1, residual_channels, kernel_size=1)
        self.upsampling = nn.ConvTranspose1d(num_mels, num_mels, kernel_size=800, stride=250, padding=150)
        self.blocks = nn.ModuleList()

        for dilation in self.dilations:
            self.blocks.append(WaveNetBlock(dilation))

        self.postprocess = nn.Sequential(nn.ReLU(),
                                         nn.Conv1d(skip_channels, skip_channels, kernel_size=1),
                                         nn.ReLU(),
                                         nn.Conv1d(skip_channels, quantized_values, kernel_size=1))

    def forward(self, x, h, training=True):
        """
        :param x: Long tensor with audio (B, T)
        :param h: Float tensor mel spec (B, num_channels, T),
        :return: Float tensor (B, T, n_quantize)
        """

        output = self.start_conv(encode_mu_law(x).unsqueeze(1))
        if training:
            h = self.upsampling(h)  # for inference we did it earlier
        skip_connections = []
        for block in self.blocks:
            output, skip = block(output, h)
            skip_connections.append(skip)

        output = sum(skip_connections)
        output = self.postprocess(output)
        return output

    def inference(self, mel):
        # first zero frame (B, receptive_field)
        waveforms = torch.zeros((mel.shape[0], self.receptive_field), dtype=torch.float).to(mel.device)
        mel = self.upsampling(mel)
        audio_len = mel.shape[2]
        for i in range(audio_len):
            x_input = waveforms[:, -self.receptive_field:]
            h_input = mel[:, :, i:i + x_input.shape[-1]]
            output = self.forward(x_input, h_input, training=False)

            quantized_audio = torch.argmax(output[:, :, -1].detach(), dim=1).unsqueeze(-1)
            waveforms = torch.cat([waveforms, decode_mu_law(dequantize(quantized_audio))], dim=1)

        return waveforms[:, -audio_len:]
