# The following code is obtained from https://github.com/echocatzh/conv-stft
# Original author: @echocatzh

import torch
from torch import nn as nn
from torch.nn import functional as F


class iSTFT(nn.Module):
    def __init__(
            self, win_len=1024, win_hop=512, fft_len=1024,
            window=None, enframe_mode='continue',
            win_sqrt=False
    ):
        """
        Implement of STFT using 1D convolution and 1D transpose convolutions.
        Implement of framing the signal in 2 ways, `break` and `continue`.
        `break` method is a kaldi-like framing.
        `continue` method is a librosa-like framing.

        More information about `perfect reconstruction`:
        1. https://ww2.mathworks.cn/help/signal/ref/stft.html
        2. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html

        Args:
            win_len (int): Number of points in one frame.  Defaults to 1024.
            win_hop (int): Number of framing stride. Defaults to 512.
            fft_len (int): Number of DFT points. Defaults to 1024.
            enframe_mode (str, optional): `break` and `continue`. Defaults to 'continue'.
            window (tensor, optional): The window tensor. Defaults to hann window.
            win_sqrt (bool, optional): using square root window. Defaults to True.
        """
        super(iSTFT, self).__init__()
        assert enframe_mode in ['break', 'continue']
        assert fft_len >= win_len
        self.win_len = win_len
        self.win_hop = win_hop
        self.fft_len = fft_len
        self.mode = enframe_mode
        self.win_sqrt = win_sqrt
        self.pad_amount = self.fft_len // 2

        if window is None:
            window = torch.hann_window(win_len)
        ifft_k, ola_k = self.__init_kernel__(window)
        self.register_buffer('ifft_k', ifft_k, persistent=False)
        self.register_buffer('ola_k', ola_k, persistent=False)

    def __init_kernel__(self, window):
        """
        Generate enframe_kernel, fft_kernel, ifft_kernel and overlap-add kernel.
        ** enframe_kernel: Using conv1d layer and identity matrix.
        ** fft_kernel: Using linear layer for matrix multiplication. In fact,
        enframe_kernel and fft_kernel can be combined, But for the sake of
        readability, I took the two apart.
        ** ifft_kernel, pinv of fft_kernel.
        ** overlap-add kernel, just like enframe_kernel, but transposed.

        Returns:
            tuple: four kernels.
        """
        tmp = torch.fft.rfft(torch.eye(self.fft_len))
        fft_kernel = torch.stack([tmp.real, tmp.imag], dim=2)
        if self.mode == 'break':
            fft_kernel = fft_kernel[:self.win_len]
        fft_kernel = torch.cat(
            (fft_kernel[:, :, 0], fft_kernel[:, :, 1]), dim=1)
        ifft_kernel = torch.pinverse(fft_kernel)[:, None, :]

        if self.mode == 'continue':
            left_pad = (self.fft_len - self.win_len) // 2
            right_pad = left_pad + (self.fft_len - self.win_len) % 2
            window = F.pad(window, (left_pad, right_pad))
        if self.win_sqrt:
            self.padded_window = window
            window = torch.sqrt(window)
        else:
            self.padded_window = window ** 2

        ifft_kernel = ifft_kernel * window
        ola_kernel = torch.eye(self.fft_len)[:self.win_len, None, :]
        if self.mode == 'continue':
            ola_kernel = torch.eye(self.fft_len)[:, None, :self.fft_len]
        return ifft_kernel, ola_kernel

    def forward(self, spec, length):
        """Call the inverse STFT (iSTFT), given tensors produced
        by the `transform` function.

        Args:
            spec (tensors): Input tensor with shape
            complex [num_batch, num_frequencies, num_frames]
            or real [num_batch, num_frequencies, num_frames, 2]
            length (int): Expected number of samples in the output audio.

        Returns:
            tensors: Reconstructed audio given magnitude and phase. Of
                shape [num_batch, num_samples]
        """
        if torch.is_complex(spec):
            real, imag = spec.real, spec.imag
        else:
            assert spec.size(-1) == 2
            real, imag = spec[..., 0], spec[..., 1]

        inputs = torch.cat([real, imag], dim=1)
        outputs = F.conv_transpose1d(inputs, self.ifft_k, stride=self.win_hop)
        t = (self.padded_window[None, :, None]).repeat(1, 1, inputs.size(-1))
        t = t.to(inputs.device)
        coff = F.conv_transpose1d(t, self.ola_k, stride=self.win_hop)
        rm_start, rm_end = self.pad_amount, self.pad_amount + length
        outputs = outputs[..., rm_start:rm_end]
        coff = coff[..., rm_start:rm_end]
        coff = torch.where(coff > 1e-8, coff, torch.ones_like(coff))
        outputs /= coff
        return outputs.squeeze(dim=1)
