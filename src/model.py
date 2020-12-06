import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import logging
import sys
from config import PostnetConfig

class OneHot(nn.Module):
    """CONVERT TO ONE-HOT VECTOR.
    Args:
        depth (int): Dimension of one-hot vector
    """

    def __init__(self, depth):
        super(OneHot, self).__init__()
        self.depth = depth

    def forward(self, x):
        """FORWARD CALCULATION.
        Arg:
            x (LongTensor): Long tensor variable with the shape (B, T).
        Returns:
            Tensor: Float tensor variable with the shape (B, depth, T).
        """
        x = x % self.depth
        x = torch.unsqueeze(x, 2)
        x_onehot = x.new_zeros(x.size(0), x.size(1), self.depth).float()

        return x_onehot.scatter_(2, x, 1)


class CausalConv1d(nn.Module):
    """1D DILATED CAUSAL CONVOLUTION."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        """FORWARD CALCULATION.
        Args:
            x (Tensor): Float tensor variable with the shape  (B, C, T).
        Returns:
            Tensor: Float tensor variable with the shape (B, C, T).
        """
        x = self.conv(x)
        if self.padding != 0:
            x = x[:, :, :-self.padding]
        return x

class UpSampling(nn.Module):
    """UPSAMPLING LAYER WITH DECONVOLUTION.
    Args:
        upsampling_factor (int): Upsampling factor.
    """

    def __init__(self, upsampling_factor, bias=True):
        super(UpSampling, self).__init__()
        self.upsampling_factor = upsampling_factor
        self.bias = bias
        self.conv = nn.ConvTranspose2d(1, 1,
                                       kernel_size=(1, self.upsampling_factor),
                                       stride=(1, self.upsampling_factor),
                                       bias=self.bias)

    def forward(self, x):
        """FORWARD CALCULATION.
        Args:
            x (Tensor): Float tensor variable with the shape (B, C, T).
        Returns:
            Tensor: Float tensor variable with the shape (B, C, T'),
                where T' = T * upsampling_factor.
        """
        x = x.unsqueeze(1)  # B x 1 x C x T
        x = self.conv(x)  # B x 1 x C x T'
        return x.squeeze(1)

class WaveNet(nn.Module):
    """CONDITIONAL WAVENET.
    Args:
        n_quantize (int): Number of quantization.
        n_aux (int): Number of aux feature dimension.
        n_resch (int): Number of filter channels for residual block.
        n_skipch (int): Number of filter channels for skip connection.
        dilation_depth (int): Number of dilation depth (e.g. if set 10, max dilation = 2^(10-1)).
        dilation_repeat (int): Number of dilation repeat.
        kernel_size (int): Filter size of dilated causal convolution.
        upsampling_factor (int): Upsampling factor.
    """

    def __init__(self, n_quantize=256, n_aux=28, n_resch=512, n_skipch=256,
                 dilation_depth=10, dilation_repeat=3, kernel_size=2, upsampling_factor=0):
        super(WaveNet, self).__init__()
        self.n_aux = n_aux
        self.n_quantize = n_quantize
        self.n_resch = n_resch
        self.n_skipch = n_skipch
        self.kernel_size = kernel_size
        self.dilation_depth = dilation_depth
        self.dilation_repeat = dilation_repeat
        self.upsampling_factor = upsampling_factor

        self.dilations = [2 ** i for i in range(self.dilation_depth)] * self.dilation_repeat
        self.receptive_field = (self.kernel_size - 1) * sum(self.dilations) + 1

        # for preprocessing
        self.onehot = OneHot(self.n_quantize)
        self.causal = CausalConv1d(self.n_quantize, self.n_resch, self.kernel_size)
        if self.upsampling_factor > 0:
            self.upsampling = UpSampling(self.upsampling_factor)

        # for residual blocks
        self.dil_sigmoid = nn.ModuleList()
        self.dil_tanh = nn.ModuleList()
        self.aux_1x1_sigmoid = nn.ModuleList()
        self.aux_1x1_tanh = nn.ModuleList()
        self.skip_1x1 = nn.ModuleList()
        self.res_1x1 = nn.ModuleList()
        for d in self.dilations:
            self.dil_sigmoid += [CausalConv1d(self.n_resch, self.n_resch, self.kernel_size, d)]
            self.dil_tanh += [CausalConv1d(self.n_resch, self.n_resch, self.kernel_size, d)]
            self.aux_1x1_sigmoid += [nn.Conv1d(self.n_aux, self.n_resch, 1)]
            self.aux_1x1_tanh += [nn.Conv1d(self.n_aux, self.n_resch, 1)]
            self.skip_1x1 += [nn.Conv1d(self.n_resch, self.n_skipch, 1)]
            self.res_1x1 += [nn.Conv1d(self.n_resch, self.n_resch, 1)]

        # for postprocessing
        self.conv_post_1 = nn.Conv1d(self.n_skipch, self.n_skipch, 1)
        self.conv_post_2 = nn.Conv1d(self.n_skipch, self.n_quantize, 1)

    def forward(self, x, h):
        """FORWARD CALCULATION.
        Args:
            x (Tensor): Long tensor variable with the shape (B, T).
            h (Tensor): Float tensor variable with the shape (B, n_aux, T),
        Returns:
            Tensor: Float tensor variable with the shape (B, T, n_quantize).
        """
        # preprocess
        output = self._preprocess(x)
        if self.upsampling_factor > 0:
            h = self.upsampling(h)

        # residual block
        skip_connections = []
        for l in range(len(self.dilations)):
            output, skip = self._residual_forward(
                output, h, self.dil_sigmoid[l], self.dil_tanh[l],
                self.aux_1x1_sigmoid[l], self.aux_1x1_tanh[l],
                self.skip_1x1[l], self.res_1x1[l])
            skip_connections.append(skip)

        # skip-connection part
        output = sum(skip_connections)
        output = self._postprocess(output)

        return output

    def generate(self, x, h, n_samples, intervals=None, mode="sampling"):
        """GENERATE WAVEFORM WITH NAIVE CALCULATION.
        Args:
            x (Tensor): Long tensor variable with the shape (1, T).
            h (Tensor): Float tensor variable with the shape (1, n_aux, n_samples + T).
            n_samples (int): Number of samples to be generated.
            intervals (int): Log interval.
            mode (str): "sampling" or "argmax".
        Returns:
            ndarray: Generated quantized wavenform (n_samples,).
        """
        # upsampling
        if self.upsampling_factor > 0:
            h = self.upsampling(h)

        # padding if the length less than receptive field size
        n_pad = self.receptive_field - x.size(1)
        if n_pad > 0:
            x = F.pad(x, (n_pad, 0), "constant", self.n_quantize // 2)
            h = F.pad(h, (n_pad, 0), "replicate")

        # generate
        samples = x[0].tolist()
        start = time.time()
        for i in range(n_samples):
            current_idx = len(samples)
            x = torch.tensor(samples[-self.receptive_field:]).long().view(1, -1)
            h_ = h[:, :, current_idx - self.receptive_field: current_idx]

            # calculate output
            output = self._preprocess(x)
            skip_connections = []
            for l in range(len(self.dilations)):
                output, skip = self._residual_forward(
                    output, h_, self.dil_sigmoid[l], self.dil_tanh[l],
                    self.aux_1x1_sigmoid[l], self.aux_1x1_tanh[l],
                    self.skip_1x1[l], self.res_1x1[l])
                skip_connections.append(skip)
            output = sum(skip_connections)
            output = self._postprocess(output)[0]  # T x n_quantize

            # get waveform
            if mode == "sampling":
                posterior = F.softmax(output[-1], dim=0)
                dist = torch.distributions.Categorical(posterior)
                sample = dist.sample()
            elif mode == "argmax":
                sample = output[-1].argmax()
            else:
                logging.error("mode should be sampling or argmax")
                sys.exit(1)
            samples.append(sample)

            # show progress
            if intervals is not None and (i + 1) % intervals == 0:
                logging.info("%d/%d estimated time = %.3f sec (%.3f sec / sample)" % (
                    i + 1, n_samples,
                    (n_samples - i - 1) * ((time.time() - start) / intervals),
                    (time.time() - start) / intervals))
                start = time.time()

        return np.array(samples[-n_samples:])

        def inference(self, x, h, n_samples, mode="argmax"):
            """GENERATE WAVEFORM WITH NAIVE CALCULATION.
            Args:
                x (Tensor): Long tensor variable with the shape (1, T).
                h (Tensor): Float tensor variable with the shape (1, n_aux, n_samples + T).
                n_samples (int): Number of samples to be generated.
                intervals (int): Log interval.
                mode (str): "sampling" or "argmax".
            Returns:
                ndarray: Generated quantized wavenform (n_samples,).
            """
            # upsampling
            h = self.upsampling(h)

            # # padding if the length less than receptive field size
            # n_pad = self.receptive_field - x.size(1)
            # if n_pad > 0:
            #     x = F.pad(x, (n_pad, 0), "constant", self.n_quantize // 2)
            #     h = F.pad(h, (n_pad, 0), "replicate")

            # generate
            samples = x[0].tolist()

            for i in range(n_samples):
                current_idx = len(samples)
                x = torch.tensor(samples[-self.receptive_field:]).long().view(1, -1)
                h_ = h[:, :, current_idx - self.receptive_field: current_idx]

                # calculate output
                output = self._preprocess(x)
                skip_connections = []
                for l in range(len(self.dilations)):
                    output, skip = self._residual_forward(
                        output, h_, self.dil_sigmoid[l], self.dil_tanh[l],
                        self.aux_1x1_sigmoid[l], self.aux_1x1_tanh[l],
                        self.skip_1x1[l], self.res_1x1[l])
                    skip_connections.append(skip)
                output = sum(skip_connections)
                output = self._postprocess(output)[0]  # T x n_quantize

                # get waveform
                if mode == "sampling":
                    posterior = F.softmax(output[-1], dim=0)
                    dist = torch.distributions.Categorical(posterior)
                    sample = dist.sample()
                elif mode == "argmax":
                    sample = output[-1].argmax()
                samples.append(sample)

            return np.array(samples[-n_samples:])

        def generate(self, h):
            zero_frame = torch.zeros((1, self.min_time)).to(h.device)
            # x - (1, n_classes, T0)
            # h - (1, aux_channels, T)
            # if self.fast_inference:
            #     for layer in self.net:
            #         layer.clear()
            h = self.upsampling(h)
            samples = h.shape[2]  # length of predicted audio signal

            # x = F.pad(x, (n_pad, 0), value=0.)
            # h = F.pad(h, (n_pad, 0), value=-11.5129251)
            output = self.start_conv(encode_mu_law(zero_frame).unsqueeze(1))
            result = []
            for _ in range(samples):
                x_input = output[:, :, -self.min_time:]
                h_input = h[:, :, -self.min_time + output.shape[-1]:output.shape[-1]]
                logprobs = self(x_input, h_input)

                sample = logprobs[0, :, -1].argmax()[None]
                result.append(sample)
                # ohe_sample = F.one_hot(sample, num_classes=self.n_classes).permute(1, 0)[None] # (1, n_classes, 1)
                output = torch.cat([output, sample], dim=-1)
            return np.array(result[-samples:])
    #        return output.argmax(dim=1)[:, n_pad:] # (1, n_classes, T) -> (1, T)
