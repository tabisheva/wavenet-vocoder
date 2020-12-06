import torch
import torch.nn as nn
import numpy as np


def encode_mu_law(x):
    # Encode audio to [-1, 1] with 256 values
    mu_law = torch.sign(x) * torch.log(256 * torch.abs(x)) / np.log(256.)
    return mu_law

def decode_mu_law(mu_law):
    # Decode mu_law
    decoded_x = np.sign(mu_law) / 255 * (torch.pow(256, torch.abs(mu_law)) - 1)
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

    def __init__(self, d):
        super(WaveNetBlock, self).__init__()

        self.dil_sigmoid = CausalConv1d(self.residual_channels, self.residual_channels, self.kernel_size, d)
        self.dil_tanh = CausalConv1d(self.residual_channels, self.residual_channels, self.kernel_size, d)
        self.mel_1x1_sigmoid = nn.Conv1d(self.num_mels, self.residual_channels, 1)
        self.mel_1x1_tanh = nn.Conv1d(self.num_mels, self.residual_channels, 1)
        self.skip_1x1 = nn.Conv1d(self.residual_channels, self.skip_channels, 1)
        self.res_1x1 = nn.Conv1d(self.residual_channels, self.residual_channels, 1)

    def forward(self, x, h):
        output_sigmoid = self.dil_sigmoid(x)
        output_tanh = self.dil_tanh(x)
        mel_output_sigmoid = self.mel_1x1_sigmoid(h)
        mel_output_tanh = self.mel_1x1_tanh(h)
        output = torch.sigmoid(output_sigmoid + mel_output_sigmoid) * torch.tanh(output_tanh + mel_output_tanh)
        skip = self.skip_1x1(output)
        output = self.res_1x1(output)
        output += x
        return output, skip

class WaveNet(nn.Module):

    def __init__(self, n_quantize=256, num_mels=80, residual_channels=120, skip_channels=240,
                 dilation_depth=8, dilation_repeat=2, kernel_size=2):
        super(WaveNet, self).__init__()
        self.num_mels = num_mels
        self.n_quantize = n_quantize
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        self.dilations = [2 ** i for i in range(dilation_depth)] * dilation_repeat

        # receptive field for causal convolutions
        self.min_time = (kernel_size - 1) * sum(self.dilations) + 1

        self.start_conv = nn.Conv1d(in_channels=1, out_channels=120, kernel_size=1)
        self.upsampling = nn.ConvTranspose1d(80, 80, kernel_size=800, stride=250, padding=150)

        self.blocks = nn.ModuleList()

        for d in self.dilations:
            self.blocks.append(d)

        self.postprocess = nn.Sequential(nn.ReLU(), nn.Conv1d(self.skip_channels, self.skip_channels, 1),
            nn.ReLU(), nn.Conv1d(self.skip_channels, self.n_quantize, 1),
        )

    def forward(self, x, h):
        """
        :param x: Long tensor with audio (B, T)
        :param h: Float tensor mel spec (B, num_channels, T),
        :return: Float tensor (B, T, n_quantize)
        """

        output = self.start_conv(encode_mu_law(x).unsqueeze(1))
        h = self.upsampling(h)

        skip_connections = []
        for block in self.blocks:
            output, skip = block(output, h)
            skip_connections.append(skip)

        output = sum(skip_connections)
        output = self.postprocess(output)

        return output