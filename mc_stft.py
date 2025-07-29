# refer:
# 【音频信号处理基本功】双通道STFT的优雅写法 - Kino的文章 - 知乎
# https://zhuanlan.zhihu.com/p/18860466841
import math
import torch
import pytorch_lightning as pl
from torch import Tensor


def get_trim_length(hop_length, min_trim=5000):
    trim_per_hop = math.ceil(min_trim / hop_length)
    trim_length = trim_per_hop * hop_length
    if trim_per_hop <= 1:
        raise ValueError(f"hop_length ({hop_length}) is too large to meet min_trim ({min_trim}) with more than one hop.")
    return trim_length


def complex_norm(spec_complex, power=1.0):
    return spec_complex.abs().pow(power)


def complex_angle(spec_complex):
    return spec_complex.angle()


def mag_phase_to_complex(mag, phase, power=1.0):
    mag_power_1 = mag.pow(1 / power)
    return mag_power_1 * torch.exp(1j * phase)


class STFT(pl.LightningModule):

    def __init__(self, n_fft, hop_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, input_signal):
        return self.to_spec_complex(input_signal)

    def to_spec_complex(self, input_signal: torch.Tensor):
        """
        input_signal: *, signal
        output: *, N, T, 2
        """
        return torch.stft(input_signal, self.n_fft, self.hop_length,
                      window=self.window.to(input_signal.device),
                      return_complex=True)

    def to_mag(self, input_signal, power=1.0):
        """
        input_signal: input signal (*, signal), power is optional
        output: *, N, T
        """
        spec_complex = self.to_spec_complex(input_signal)
        return complex_norm(spec_complex, power)

    def to_phase(self, input_signal):
        """
        input_signal: *, signal
        output: *, N, T
        """
        spec_complex = self.to_spec_complex(input_signal)
        return complex_angle(spec_complex)

    def to_mag_phase(self, input_signal, power=1.0):
        """
        input_signal: input signal (*, signal), power is optional
        output: tuple (mag(*, N, T) , phase(*, N, T))
        """
        spec_complex = self.to_spec_complex(input_signal)
        return complex_norm(spec_complex, power), complex_angle(spec_complex)

    def restore_complex(self, spec_complex):
        """
        spec_complex: (*, F, T) - complex tensor
        returns: waveform (*, T)
        """
        return torch.istft(spec_complex, self.n_fft, self.hop_length,
                        window=self.window.to(spec_complex.device))

    def restore_mag_phase(self, mag, phase, power=1.):
        """
        input_signal: mag(*, N, T), phase(*, N, T), power is optional
        output: *, signal
        """
        spec_complex = mag_phase_to_complex(mag, phase, power)
        return self.restore_complex(spec_complex)


class multi_channeled_STFT(pl.LightningModule):
    def __init__(self, n_fft, hop_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.stft = STFT(n_fft, hop_length)

    def forward(self, input_signal):
        return self.to_spec_complex(input_signal)

    def to_spec_complex(self, input_signal) -> Tensor:
        """
        input_signal: [B, C, T]
        output: [B, C, F, T']  # complex dtype
        """
        B, C, T = input_signal.shape
        specs = [self.stft.to_spec_complex(input_signal[:, ch]) for ch in range(C)]
        return torch.stack(specs, dim=1)

    def to_mag(self, input_signal, power=1.0):
        """
        input_signal: [B, C, T]
        output: [B, C, F, T']
        """
        B, C, T = input_signal.shape
        mags = [self.stft.to_mag(input_signal[:, ch], power) for ch in range(C)]
        return torch.stack(mags, dim=1)

    def to_phase(self, input_signal):
        """
        input_signal: [B, C, T]
        output: [B, C, F, T']
        """
        B, C, T = input_signal.shape
        phases = [self.stft.to_phase(input_signal[:, ch]) for ch in range(C)]
        return torch.stack(phases, dim=1)

    def to_mag_phase(self, input_signal, power=1.0):
        """
        input_signal: [B, C, T]
        output: (mag: [B, C, F, T'], phase: [B, C, F, T'])
        """
        B, C, T = input_signal.shape
        mags, phases = [], []
        for ch in range(C):
            mag, phase = self.stft.to_mag_phase(input_signal[:, ch], power)
            mags.append(mag)
            phases.append(phase)
        return torch.stack(mags, dim=1), torch.stack(phases, dim=1)

    def restore_complex(self, spec_complex):
        """
        spec_complex: [B, C, F, T] (complex dtype)
        return: [B, C, T]
        """
        B, C = spec_complex.shape[:2]
        signals = [self.stft.restore_complex(spec_complex[:, ch]) for ch in range(C)]
        return torch.stack(signals, dim=1)

    def restore_mag_phase(self, mag, phase, power=1.):
        """
        input_signal: mag, phase: [B, C, F, T']
        output: [B, C, T]
        """
        B, C, F, T_ = mag.shape
        signals = [self.stft.restore_mag_phase(mag[:, ch], phase[:, ch], power) for ch in range(C)]
        return torch.stack(signals, dim=1)
    

if __name__ == "__main__":
    B, C, T = 16, 2, 16000
    wav = torch.rand(B, C, T)

    stft_module = multi_channeled_STFT(n_fft=512, hop_length=128)

    # STFT: [B, C, F, T']
    spec = stft_module.to_spec_complex(wav) # STFT shape: torch.Size([16, 2, 257, 126]) torch.complex64
    print("STFT shape:", spec.shape, spec.dtype)

    # ISTFT: [B, C, T]
    recon = stft_module.restore_complex(spec)

    print("Reconstructed waveform shape:", recon.shape) # Reconstructed waveform shape: torch.Size([16, 2, 16000])
    print("Match original shape:", recon.shape == wav.shape) # Match original shape: True
    print("Exactly equal:", torch.equal(recon, wav)) # Exactly equal: False
    print("Allclose:", torch.allclose(recon, wav, atol=1e-5)) # Allclose: True
