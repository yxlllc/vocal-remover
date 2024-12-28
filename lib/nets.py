import torch
from torch import nn
import torch.nn.functional as F

from . import layers


class BaseNet(nn.Module):

    def __init__(self, nin, nout, nin_lstm, nout_lstm, dilations=((4, 2), (8, 4), (12, 6))):
        super(BaseNet, self).__init__()
        self.enc1 = layers.Conv2DBNActiv(nin, nout, 3, 1, 1)
        self.enc2 = layers.Encoder(nout, nout * 2, 3, 2, 1)
        self.enc3 = layers.Encoder(nout * 2, nout * 4, 3, 2, 1)
        self.enc4 = layers.Encoder(nout * 4, nout * 6, 3, 2, 1)
        self.enc5 = layers.Encoder(nout * 6, nout * 8, 3, 2, 1)

        self.aspp = layers.ASPPModule(nout * 8, nout * 8, dilations, dropout=True)

        self.dec4 = layers.Decoder(nout * (6 + 8), nout * 6, 3, 1, 1)
        self.dec3 = layers.Decoder(nout * (4 + 6), nout * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(nout * (2 + 4), nout * 2, 3, 1, 1)
        self.lstm_dec2 = layers.LSTMModule(nout * 2, nin_lstm, nout_lstm)
        self.dec1 = layers.Decoder(nout * (1 + 2) + 1, nout * 1, 3, 1, 1)

    def __call__(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        h = self.aspp(e5)

        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = torch.cat([h, self.lstm_dec2(h)], dim=1)
        h = self.dec1(h, e1)

        return h


class CascadedNet(nn.Module):

    def __init__(self, n_fft, hop_length, nout=32, nout_lstm=128, is_complex=False, is_mono=False):
        super(CascadedNet, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.seg_length = 32 * hop_length
        self.is_complex = is_complex
        self.is_mono = is_mono
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.nin_lstm = self.max_bin // 2
        self.offset = 64

        nin = 4 if is_complex else 2
        if is_mono:
            nin = nin // 2


        self.stg1_full_band_net = BaseNet(
            nin, nout // 2, self.nin_lstm, nout_lstm
        )

        self.stg2_full_band_net = BaseNet(
            nout // 2 + nin, nout // 2, self.nin_lstm, nout_lstm
        )

        self.stg3_full_band_net = BaseNet(
            nout + nin, nout, self.nin_lstm, nout_lstm
        )

        self.out = nn.Conv2d(nout, 3 * nin, 1, bias=False)

    def forward(self, x, hb_th=0.0, he_th=0.0):
        if self.is_complex:
            x = torch.cat([x.real, x.imag], dim=1)

        x = x[:, :, :self.max_bin]
         
        x = torch.cat((x, self.stg1_full_band_net(x)), dim=1)
        x = torch.cat((x, self.stg2_full_band_net(x)), dim=1)
        x = self.stg3_full_band_net(x)

        if self.is_complex:
            mask = self.out(x)
            if self.is_mono:
                mask = torch.complex(mask[:, :3], mask[:, 3:])
            else:
                mask = torch.complex(mask[:, :6], mask[:, 6:])
            mask = self.bounded_mask(mask)
        else:
            mask = torch.sigmoid(self.out(x))
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode='replicate'
        )
        mask_h, mask_hb, mask_he = torch.split(mask, mask.size(1) // 3, dim=1)
        if hb_th > 0:
            mask_hb[mask_hb.abs() < hb_th] = 0
        if he_th > 0:
            mask_he[mask_he.abs() < he_th] = 0
        mask_hb = mask_h * mask_hb
        mask_he = mask_h * mask_he
        return mask_h, mask_hb, mask_he

    def bounded_mask(self, mask, eps=1e-8):
        mask_mag = torch.abs(mask)
        mask = torch.tanh(mask_mag) * mask / (mask_mag + eps)
        return mask

    def audio2spec(self, x, use_pad=False):
        B, C, T = x.shape
        x = x.reshape(B * C, T)
        if use_pad:
            T1 = T + self.hop_length
            T_pad = self.seg_length * ((T1 - 1) // self.seg_length + 1) - T1
            nl_pad = T_pad // 2 // self.hop_length
            Tl_pad = nl_pad * self.hop_length
            x = F.pad(x, (Tl_pad, T_pad - Tl_pad))
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
            window=self.window,
            pad_mode='constant'
        )
        spec = spec.reshape(B, C, spec.shape[-2], spec.shape[-1])
        return spec

    def spec2audio(self, x):
        B, C, N, T = x.shape
        x = x.reshape(-1, N, T)
        x = torch.istft(x, self.n_fft, self.hop_length, window=self.window)
        x = x.reshape(B, C, -1)
        return x

    def predict_fromaudio(self, x, return_h=True, return_hb=True, return_he=True, hb_th=0.0, he_th=0.0):
        B, C, T = x.shape
        x = x.reshape(B * C, T)
        T1 = T + self.hop_length
        T_pad = self.seg_length * ((T1 - 1) // self.seg_length + 1) - T1       
        nl_pad = T_pad // 2 // self.hop_length
        Tl_pad = nl_pad * self.hop_length
        x = F.pad(x, (Tl_pad, T_pad - Tl_pad))
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
            window=self.window,
            pad_mode='constant'
        )
        spec = spec.reshape(B, C, spec.shape[-2], spec.shape[-1])
        
        mask_h, mask_hb, mask_he = self.forward(spec, hb_th=hb_th, he_th=he_th)
        
        if return_h:
            spec_h = spec * mask_h
            spec_h = spec_h.reshape(B * C, spec.shape[-2], spec.shape[-1])
            x_h = torch.istft(spec_h, self.n_fft, self.hop_length, window=self.window)
            x_h = x_h[:, Tl_pad: Tl_pad + T]
            x_h = x_h.reshape(B, C, T)
        else:
            x_h = None
            
        if return_hb:
            spec_hb = spec * mask_hb
            spec_hb = spec_hb.reshape(B * C, spec.shape[-2], spec.shape[-1])
            x_hb = torch.istft(spec_hb, self.n_fft, self.hop_length, window=self.window)
            x_hb = x_hb[:, Tl_pad: Tl_pad + T]
            x_hb = x_hb.reshape(B, C, T)
        else:
            x_hb = None
            
        if return_he:
            spec_he = spec * mask_he
            spec_he = spec_he.reshape(B * C, spec.shape[-2], spec.shape[-1])
            x_he = torch.istft(spec_he, self.n_fft, self.hop_length, window=self.window)
            x_he = x_he[:, Tl_pad: Tl_pad + T]
            x_he = x_he.reshape(B, C, T)
        else:
            x_he = None
               
        return x_h, x_hb, x_he
