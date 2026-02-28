"""
Stage 1: 音频基础
=================

学习目标:
  1. 理解数字音频的基本概念：采样率、位深度、声道
  2. 掌握音频信号的时域与频域表示
  3. 理解短时傅里叶变换 (STFT) 及其逆变换 (ISTFT)
  4. 理解梅尔频谱 (Mel Spectrogram) 的原理

为什么这是第一步？
  Neural Audio Codec 本质上是对音频信号进行压缩和重建。
  要理解它，首先必须理解音频信号本身的数学表示。
"""

import torch
import torchaudio
import numpy as np


# =============================================================================
# 1. 生成基本音频信号
# =============================================================================

def generate_sine_wave(
    freq_hz: float = 440.0,
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> torch.Tensor:
    """
    生成正弦波。这是最基本的音频信号——单一频率的纯音。

    Args:
        freq_hz:    频率(Hz)，440Hz 是标准 A 音
        duration_s: 时长(秒)
        sample_rate: 采样率，每秒采样点数。16kHz 常用于语音，44.1kHz 用于音乐
        amplitude:  振幅，控制音量大小

    Returns:
        shape (1, num_samples) 的张量
    """
    t = torch.linspace(0, duration_s, int(sample_rate * duration_s))
    waveform = amplitude * torch.sin(2 * torch.pi * freq_hz * t)
    return waveform.unsqueeze(0)  # (1, T) — 单声道


def generate_composite_signal(
    freqs: list[float],
    duration_s: float = 1.0,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    将多个正弦波叠加，模拟真实音频中的谐波结构。
    真实乐器的声音就是由基频和多个谐波叠加而成的。
    """
    t = torch.linspace(0, duration_s, int(sample_rate * duration_s))
    waveform = torch.zeros_like(t)
    for i, f in enumerate(freqs):
        waveform += (1.0 / (i + 1)) * torch.sin(2 * torch.pi * f * t)
    waveform = waveform / waveform.abs().max()  # 归一化到 [-1, 1]
    return waveform.unsqueeze(0)


# =============================================================================
# 2. 短时傅里叶变换 (STFT)
# =============================================================================

class STFTAnalyzer:
    """
    STFT 是 Neural Audio Codec 中最核心的信号分析工具。

    核心思想：将长信号切成短帧(通常 20-40ms)，对每帧做 FFT，
    得到随时间变化的频谱信息 —— 即"时频表示"。

    关键参数：
      - n_fft:      FFT 点数，决定频率分辨率。越大频率分辨率越高，但时间分辨率越低
      - hop_length:  帧移，相邻帧之间移动的采样点数。决定时间分辨率
      - win_length:  窗长，每帧的长度
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)

    def stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        计算 STFT。

        输入: waveform (B, T) 或 (1, T)
        输出: complex tensor (B, F, T') 其中 F = n_fft//2 + 1
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        window = self.window.to(waveform.device)
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )

    def istft(self, spec: torch.Tensor) -> torch.Tensor:
        """
        逆 STFT：从频谱重建波形。
        这证明了 STFT 是可逆的（在合理的参数设置下）——
        这也是为什么 Neural Audio Codec 可以在频域工作的基础。
        """
        window = self.window.to(spec.device)
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
        )

    def magnitude(self, spec: torch.Tensor) -> torch.Tensor:
        """幅度谱：|STFT(x)|，丢弃相位信息"""
        return spec.abs()

    def phase(self, spec: torch.Tensor) -> torch.Tensor:
        """相位谱：angle(STFT(x))"""
        return spec.angle()

    def log_magnitude(self, spec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        对数幅度谱。人耳对响度的感知是对数的，
        因此在损失函数中常用 log 幅度谱。
        """
        return torch.log(spec.abs() + eps)


# =============================================================================
# 3. 梅尔频谱
# =============================================================================

class MelSpectrogramExtractor:
    """
    梅尔频谱是模拟人耳听觉特性的频谱表示。

    人耳对低频更敏感，对高频区分度较低。梅尔刻度通过非线性映射
    将频率轴压缩，使得低频部分有更高的分辨率。

    在 Neural Audio Codec 中，梅尔频谱常用于：
      1. 感知损失函数（重建质量度量）
      2. 特征提取（判别器输入）
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
    ):
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.transform(waveform)

    def log_mel(self, waveform: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """对数梅尔频谱，常用于语音相关任务"""
        mel = self.transform(waveform)
        return torch.log(mel + eps)


# =============================================================================
# 4. 演示：验证 STFT 的可逆性
# =============================================================================

def demo_stft_invertibility():
    """
    验证 STFT -> ISTFT 的重建精度。
    这个性质对于 Audio Codec 至关重要：我们需要能从频域表示完美重建时域信号。
    """
    sr = 16000
    waveform = generate_composite_signal([220, 440, 880], duration_s=0.5, sample_rate=sr)

    analyzer = STFTAnalyzer(n_fft=1024, hop_length=256)
    spec = analyzer.stft(waveform)
    reconstructed = analyzer.istft(spec)

    min_len = min(waveform.shape[-1], reconstructed.shape[-1])
    error = (waveform[..., :min_len] - reconstructed[..., :min_len]).abs().max().item()
    print(f"STFT 可逆性验证 — 最大重建误差: {error:.2e}")
    assert error < 1e-5, "STFT 重建误差过大！"
    print("通过！STFT 是可逆变换。\n")
    return error


def demo_compression_ratio():
    """
    展示 STFT 表示的数据量 vs 原始波形。
    Neural Audio Codec 的目标就是在保持质量的前提下进一步压缩。
    """
    sr = 16000
    duration = 1.0
    waveform = generate_sine_wave(440, duration, sr)
    analyzer = STFTAnalyzer(n_fft=1024, hop_length=256)
    spec = analyzer.stft(waveform)

    raw_size = waveform.numel()
    spec_size = spec.numel() * 2  # complex = 2x float

    print(f"原始波形采样点数: {raw_size}")
    print(f"STFT 频谱实数个数: {spec_size}")
    print(f"STFT 表示/原始 比率: {spec_size / raw_size:.2f}x")
    print("(STFT 本身不压缩数据，它只是改变了表示方式)\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 1: 音频基础")
    print("=" * 60)

    print("\n--- 1. 生成正弦波 ---")
    wave = generate_sine_wave(440, 0.5, 16000)
    print(f"生成 440Hz 正弦波: shape={wave.shape}, 范围=[{wave.min():.2f}, {wave.max():.2f}]")

    print("\n--- 2. 复合信号 ---")
    composite = generate_composite_signal([220, 440, 880], 0.5, 16000)
    print(f"复合信号: shape={composite.shape}")

    print("\n--- 3. STFT 分析 ---")
    analyzer = STFTAnalyzer()
    spec = analyzer.stft(composite)
    print(f"STFT 输出: shape={spec.shape}  (频率bins x 时间帧)")
    print(f"幅度谱 范围: [{analyzer.magnitude(spec).min():.4f}, {analyzer.magnitude(spec).max():.4f}]")

    print("\n--- 4. STFT 可逆性验证 ---")
    demo_stft_invertibility()

    print("--- 5. 数据量对比 ---")
    demo_compression_ratio()

    print("\n--- 6. 梅尔频谱 ---")
    mel_extractor = MelSpectrogramExtractor()
    mel = mel_extractor(composite)
    log_mel = mel_extractor.log_mel(composite)
    print(f"梅尔频谱: shape={mel.shape}  (n_mels x 时间帧)")
    print(f"对数梅尔频谱: shape={log_mel.shape}")

    print("\n✓ Stage 1 完成！你已掌握了音频信号的基本表示方法。")
    print("  下一步 → Stage 2: 构建简单的音频自编码器")
