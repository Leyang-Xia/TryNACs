"""
Stage 7: 感知损失函数 (Perceptual Losses)
==========================================

学习目标:
  1. 理解为什么需要多种损失函数
  2. 实现多分辨率 STFT 损失 (Multi-Resolution STFT Loss)
  3. 实现梅尔频谱损失
  4. 理解各损失函数的互补作用

损失函数体系全景:

  Neural Audio Codec 的总损失 = 加权组合:

  ┌─ 时域损失 ──── L1 波形损失: 基本的样本级重建
  │
  ├─ 频域损失 ──── 多分辨率 STFT 损失: 不同时频分辨率的频谱匹配
  │                梅尔频谱损失: 基于人耳感知的频谱匹配
  │
  ├─ 对抗损失 ──── 生成器损失: 欺骗判别器
  │                Feature Matching: 匹配判别器中间特征
  │
  └─ 量化损失 ──── Commitment Loss: 稳定 VQ 训练

  为什么需要这么多？
    - 时域损失: 确保整体波形形状正确
    - 频域损失: 确保频谱结构正确（人耳更敏感于频谱）
    - 对抗损失: 确保音频听起来自然（非模糊）
    - 量化损失: 确保码本训练稳定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. STFT 损失
# =============================================================================

class STFTLoss(nn.Module):
    """
    单分辨率 STFT 损失。

    包含两部分：
      1. 频谱收敛 (Spectral Convergence): 归一化的 Frobenius 范数差
         → 衡量整体频谱形状的匹配度
      2. 对数幅度损失 (Log Magnitude): L1 距离
         → 衡量对数刻度上的频谱细节匹配度

    为什么用对数？因为人耳对响度的感知是对数的。
    安静的细节和响亮的主体在对数域中被等同对待。
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length))

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        return torch.stft(
            x, self.n_fft, self.hop_length, self.win_length,
            window=self.window, return_complex=True,
        ).abs()

    def forward(
        self, x_pred: torch.Tensor, x_real: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_pred: (B, 1, T) 或 (B, T) 重建波形
            x_real: (B, 1, T) 或 (B, T) 原始波形

        Returns:
            sc_loss: 频谱收敛损失
            mag_loss: 对数幅度损失
        """
        mag_pred = self._stft(x_pred)
        mag_real = self._stft(x_real)

        sc_loss = torch.norm(mag_real - mag_pred, p='fro') / (torch.norm(mag_real, p='fro') + 1e-8)
        mag_loss = F.l1_loss(torch.log(mag_pred + 1e-8), torch.log(mag_real + 1e-8))

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """
    多分辨率 STFT 损失 — Neural Audio Codec 的标配。

    使用多组 STFT 参数，捕捉不同的时频权衡：
      - 大 FFT (2048): 高频率分辨率 → 精确的基频和谐波
      - 中 FFT (1024): 平衡
      - 小 FFT (256):  高时间分辨率 → 精确的瞬态（鼓击、辅音）

    这确保重建音频在所有时频分辨率上都与原始音频匹配。
    """

    def __init__(
        self,
        fft_sizes: list[int] = None,
        hop_sizes: list[int] = None,
        win_sizes: list[int] = None,
    ):
        super().__init__()
        fft_sizes = fft_sizes or [2048, 1024, 512, 256]
        hop_sizes = hop_sizes or [512, 256, 128, 64]
        win_sizes = win_sizes or [2048, 1024, 512, 256]

        self.stft_losses = nn.ModuleList([
            STFTLoss(n_fft=n, hop_length=h, win_length=w)
            for n, h, w in zip(fft_sizes, hop_sizes, win_sizes)
        ])

    def forward(
        self, x_pred: torch.Tensor, x_real: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回所有分辨率的平均 SC 和 Mag 损失"""
        sc_total, mag_total = 0.0, 0.0
        for stft_loss in self.stft_losses:
            sc, mag = stft_loss(x_pred, x_real)
            sc_total += sc
            mag_total += mag
        n = len(self.stft_losses)
        return sc_total / n, mag_total / n


# =============================================================================
# 2. 梅尔频谱损失
# =============================================================================

class MelSpectrogramLoss(nn.Module):
    """
    梅尔频谱损失。

    与 STFT 损失的区别：
      - STFT 使用线性频率刻度
      - 梅尔使用对数频率刻度（模拟人耳）

    人耳对低频区域更敏感，梅尔刻度放大了低频区域的权重。
    这使得梅尔损失更关注对听觉重要的频率成分。

    在 EnCodec 中，梅尔损失是多个分辨率的组合，
    权重随着 STFT 窗口大小呈 2 的幂次关系。
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        log_scale: bool = True,
    ):
        super().__init__()
        import torchaudio
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=1.0,  # 幅度谱而非功率谱
        )
        self.log_scale = log_scale

    def forward(self, x_pred: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
        mel_pred = self.mel_transform(x_pred)
        mel_real = self.mel_transform(x_real)

        if self.log_scale:
            mel_pred = torch.log(mel_pred.clamp(min=1e-5))
            mel_real = torch.log(mel_real.clamp(min=1e-5))

        return F.l1_loss(mel_pred, mel_real)


class MultiScaleMelLoss(nn.Module):
    """
    多尺度梅尔频谱损失 (EnCodec 使用)。

    使用多组 mel 参数，覆盖不同的时频分辨率。
    每组的权重按 2^i 递增（大窗口=长程结构，权重更高）。
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        fft_sizes: list[int] = None,
        hop_sizes: list[int] = None,
        n_mels_list: list[int] = None,
    ):
        super().__init__()
        fft_sizes = fft_sizes or [32, 64, 128, 256, 512, 1024, 2048]
        hop_sizes = hop_sizes or [s // 4 for s in fft_sizes]
        n_mels_list = n_mels_list or [5, 10, 20, 40, 80, 80, 80]

        self.mel_losses = nn.ModuleList()
        self.weights = []
        for i, (n_fft, hop, n_mels) in enumerate(zip(fft_sizes, hop_sizes, n_mels_list)):
            self.mel_losses.append(MelSpectrogramLoss(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop,
                n_mels=n_mels,
            ))
            self.weights.append(2 ** i)

    def forward(self, x_pred: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
        total = 0.0
        for loss_fn, w in zip(self.mel_losses, self.weights):
            total += w * loss_fn(x_pred, x_real)
        return total / sum(self.weights)


# =============================================================================
# 3. 时域损失
# =============================================================================

class WaveformLoss(nn.Module):
    """
    简单的时域 L1/L2 损失。

    虽然简单，但它确保了整体波形形状的正确性。
    L1 比 L2 更鲁棒（对异常值不那么敏感）。
    """

    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        self.loss_fn = F.l1_loss if loss_type == 'l1' else F.mse_loss

    def forward(self, x_pred: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(x_pred, x_real)


# =============================================================================
# 4. 总损失组合器
# =============================================================================

class CodecLossBalancer(nn.Module):
    """
    损失平衡器 — 将所有损失按权重组合。

    典型的 EnCodec 权重配置:
      - 时域 L1:        λ_t = 0.1
      - 多分辨率 STFT:   λ_f = 1.0
      - 梅尔损失:        λ_mel = 1.0
      - 对抗损失:        λ_adv = 1.0 / num_discriminators
      - Feature Matching: λ_fm = 2.0
      - VQ Commitment:    λ_commit = 1.0
    """

    def __init__(
        self,
        lambda_time: float = 0.1,
        lambda_freq: float = 1.0,
        lambda_mel: float = 1.0,
        lambda_adv: float = 0.1,
        lambda_feat: float = 2.0,
        lambda_commit: float = 1.0,
    ):
        super().__init__()
        self.weights = {
            'time': lambda_time,
            'freq_sc': lambda_freq,
            'freq_mag': lambda_freq,
            'mel': lambda_mel,
            'adv': lambda_adv,
            'feat_match': lambda_feat,
            'commitment': lambda_commit,
        }

    def forward(self, losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        total = 0.0
        weighted = {}
        for key, loss in losses.items():
            w = self.weights.get(key, 1.0)
            weighted[key] = w * loss
            total += weighted[key]
        return total, weighted


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 7: 感知损失函数")
    print("=" * 60)

    x_real = torch.randn(2, 1, 16000)
    x_pred = x_real + 0.1 * torch.randn_like(x_real)  # 轻微噪声
    x_bad = torch.randn(2, 1, 16000)  # 完全不同

    print("\n--- 1. 单分辨率 STFT 损失 ---")
    stft_loss = STFTLoss(n_fft=1024, hop_length=256)
    sc_good, mag_good = stft_loss(x_pred, x_real)
    sc_bad, mag_bad = stft_loss(x_bad, x_real)
    print(f"  轻微噪声: SC={sc_good.item():.4f}, Mag={mag_good.item():.4f}")
    print(f"  完全不同:  SC={sc_bad.item():.4f}, Mag={mag_bad.item():.4f}")
    print(f"  → 损失能有效区分好坏重建")

    print("\n--- 2. 多分辨率 STFT 损失 ---")
    mr_stft = MultiResolutionSTFTLoss()
    sc_mr, mag_mr = mr_stft(x_pred, x_real)
    print(f"  多分辨率 SC:  {sc_mr.item():.4f}")
    print(f"  多分辨率 Mag: {mag_mr.item():.4f}")
    print(f"  包含 {len(mr_stft.stft_losses)} 个不同分辨率")

    print("\n--- 3. 梅尔频谱损失 ---")
    mel_loss = MelSpectrogramLoss(sample_rate=16000)
    mel_good = mel_loss(x_pred, x_real)
    mel_bad = mel_loss(x_bad, x_real)
    print(f"  轻微噪声: {mel_good.item():.4f}")
    print(f"  完全不同:  {mel_bad.item():.4f}")

    print("\n--- 4. 多尺度梅尔损失 ---")
    ms_mel = MultiScaleMelLoss(sample_rate=16000)
    ms_mel_v = ms_mel(x_pred, x_real)
    print(f"  多尺度梅尔损失: {ms_mel_v.item():.4f}")
    print(f"  包含 {len(ms_mel.mel_losses)} 个不同尺度")

    print("\n--- 5. 时域损失 ---")
    wav_loss = WaveformLoss('l1')
    l1_good = wav_loss(x_pred, x_real)
    l1_bad = wav_loss(x_bad, x_real)
    print(f"  L1 轻微噪声: {l1_good.item():.4f}")
    print(f"  L1 完全不同:  {l1_bad.item():.4f}")

    print("\n--- 6. 损失组合 ---")
    balancer = CodecLossBalancer()
    losses = {
        'time': l1_good,
        'freq_sc': sc_good,
        'freq_mag': mag_good,
        'mel': mel_good,
    }
    total, weighted = balancer(losses)
    print(f"  各项加权损失:")
    for k, v in weighted.items():
        print(f"    {k:>12}: {v.item():.4f}")
    print(f"  总损失: {total.item():.4f}")

    print("\n✓ Stage 7 完成！")
    print("  关键理解：多种损失从不同角度衡量重建质量——")
    print("  时域波形、频域频谱、人耳感知梅尔、对抗真实感。")
    print("  它们的加权组合驱动模型在各个维度上优化。")
    print("  下一步 → Stage 8: 整合所有组件为完整的 Neural Audio Codec！")
