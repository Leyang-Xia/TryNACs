"""
Stage 8: 完整的 Neural Audio Codec
====================================

这是最终的整合阶段，将前 7 个阶段的所有组件组合成一个完整的系统。

整体架构:

  ┌──────────────────────────────────────────────────────┐
  │                  Neural Audio Codec                   │
  │                                                       │
  │  编码 (Compress):                                     │
  │    波形 (24kHz) ──→ SEANet Encoder ──→ RVQ ──→ 码本索引│
  │    [1, 1, 24000]    [1, 128, 75]     [1, 8, 75]      │
  │                                                       │
  │  解码 (Decompress):                                   │
  │    码本索引 ──→ RVQ Lookup ──→ SEANet Decoder ──→ 波形  │
  │    [1, 8, 75]   [1, 128, 75]   [1, 1, 24000]         │
  │                                                       │
  │  训练:                                                 │
  │    Generator: Encoder + RVQ + Decoder                 │
  │    Discriminator: MPD + MSD                           │
  │    Loss: 时域 + 频域 + 对抗 + Feature Match + VQ      │
  └──────────────────────────────────────────────────────┘

  比特率计算 (24kHz, 8层RVQ, 1024码本, hop=320):
    = 8 × log2(1024) × (24000/320)
    = 8 × 10 × 75
    = 6000 bps = 6 kbps

参考论文:
  - SoundStream (2021): https://arxiv.org/abs/2107.03312
  - EnCodec (2022): https://arxiv.org/abs/2210.13438
  - DAC (2023): https://arxiv.org/abs/2306.06546
"""

import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stage4_residual_vq.residual_vq import ResidualVectorQuantizer
from stage5_conv_encdec.encoder_decoder import SEANetEncoder, SEANetDecoder
from stage6_discriminator.discriminators import (
    MultiPeriodDiscriminator, MultiScaleDiscriminator,
    adversarial_generator_loss, adversarial_discriminator_loss,
    feature_matching_loss,
)
from stage7_losses.perceptual_losses import (
    MultiResolutionSTFTLoss, MultiScaleMelLoss, WaveformLoss,
)


class NeuralAudioCodec(nn.Module):
    """
    完整的 Neural Audio Codec。

    将前面所有阶段的组件整合:
      Stage 5 → SEANet Encoder/Decoder
      Stage 4 → Residual Vector Quantizer
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        # 编码器/解码器参数
        base_channels: int = 32,
        code_dim: int = 128,
        encoder_strides: tuple = (2, 4, 5, 8),
        causal: bool = True,
        # RVQ 参数
        num_quantizers: int = 8,
        num_codes: int = 1024,
        commitment_weight: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self.hop_length = math.prod(encoder_strides)
        self.num_quantizers = num_quantizers
        self.num_codes = num_codes

        self.encoder = SEANetEncoder(
            in_channels=channels,
            base_channels=base_channels,
            code_dim=code_dim,
            strides=encoder_strides,
            causal=causal,
        )

        self.quantizer = ResidualVectorQuantizer(
            num_quantizers=num_quantizers,
            num_codes=num_codes,
            code_dim=code_dim,
            commitment_weight=commitment_weight,
        )

        self.decoder = SEANetDecoder(
            out_channels=channels,
            base_channels=base_channels,
            code_dim=code_dim,
            strides=encoder_strides,
            causal=causal,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码：波形 → 码本索引。

        Args:
            x: (B, 1, T) 波形
        Returns:
            codes: (B, num_quantizers, T') 整数码本索引
        """
        z = self.encoder(x)
        _, codes, _ = self.quantizer(z)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        解码：码本索引 → 波形。

        Args:
            codes: (B, num_quantizers, T')
        Returns:
            x: (B, 1, T) 重建波形
        """
        z_q = self.quantizer.decode_from_indices(codes)
        return self.decoder(z_q)

    def forward(
        self,
        x: torch.Tensor,
        num_quantizers: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        前向传播（训练用）。

        Args:
            x: (B, 1, T) 输入波形
            num_quantizers: 可选，使用的 RVQ 层数

        Returns:
            x_recon: (B, 1, T) 重建波形
            codes: (B, N, T') 码本索引
            vq_losses: dict VQ 相关损失
        """
        z = self.encoder(x)
        z_q, codes, vq_losses = self.quantizer(z, num_quantizers=num_quantizers)
        x_recon = self.decoder(z_q)
        return x_recon, codes, vq_losses

    def bitrate(self, num_quantizers: int | None = None) -> float:
        """计算比特率 (bps)"""
        n_q = num_quantizers or self.num_quantizers
        bits_per_code = math.log2(self.num_codes)
        frames_per_second = self.sample_rate / self.hop_length
        return n_q * bits_per_code * frames_per_second

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> dict:
        """
        压缩接口 — 模拟实际的编码传输。

        Returns:
            dict with:
                codes: 整数码本索引
                original_length: 原始波形长度
                metadata: 解码所需的元信息
        """
        original_length = x.shape[-1]
        pad = (self.hop_length - x.shape[-1] % self.hop_length) % self.hop_length
        if pad > 0:
            x = F.pad(x, (0, pad))
        codes = self.encode(x)
        return {
            'codes': codes,
            'original_length': original_length,
        }

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        """
        解压缩接口 — 从码本索引重建波形。
        """
        x_recon = self.decode(compressed['codes'])
        return x_recon[..., :compressed['original_length']]


# =============================================================================
# 训练器
# =============================================================================

class CodecTrainer:
    """
    Neural Audio Codec 训练器。

    训练采用交替优化策略:
      1. 固定 Generator，训练 Discriminator（真假分类）
      2. 固定 Discriminator，训练 Generator（重建 + 欺骗判别器）

    这与 GAN 的训练方式一致。
    """

    def __init__(
        self,
        codec: NeuralAudioCodec,
        sample_rate: int = 24000,
        lr_g: float = 3e-4,
        lr_d: float = 3e-4,
        lambda_time: float = 0.1,
        lambda_freq: float = 1.0,
        lambda_mel: float = 1.0,
        lambda_adv: float = 0.1,
        lambda_feat: float = 2.0,
        lambda_commit: float = 1.0,
    ):
        self.codec = codec
        self.lambda_time = lambda_time
        self.lambda_freq = lambda_freq
        self.lambda_mel = lambda_mel
        self.lambda_adv = lambda_adv
        self.lambda_feat = lambda_feat
        self.lambda_commit = lambda_commit

        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator(num_scales=3)

        self.mr_stft_loss = MultiResolutionSTFTLoss()
        self.mel_loss = MultiScaleMelLoss(sample_rate=sample_rate)
        self.wav_loss = WaveformLoss('l1')

        self.opt_g = torch.optim.AdamW(
            codec.parameters(), lr=lr_g, betas=(0.5, 0.9)
        )
        self.opt_d = torch.optim.AdamW(
            list(self.mpd.parameters()) + list(self.msd.parameters()),
            lr=lr_d, betas=(0.5, 0.9),
        )

    def train_step(
        self, x_real: torch.Tensor
    ) -> dict[str, float]:
        """
        单步训练。

        Args:
            x_real: (B, 1, T) 真实波形

        Returns:
            losses: 各项损失值的字典
        """
        # === Generator 前向 ===
        x_recon, codes, vq_losses = self.codec(x_real)
        min_len = min(x_real.shape[-1], x_recon.shape[-1])
        x_real_t = x_real[..., :min_len]
        x_recon_t = x_recon[..., :min_len]

        # === Step 1: 训练判别器 ===
        self.opt_d.zero_grad()

        scores_real_p, _ = self.mpd(x_real_t)
        scores_fake_p, _ = self.mpd(x_recon_t.detach())
        scores_real_s, _ = self.msd(x_real_t)
        scores_fake_s, _ = self.msd(x_recon_t.detach())

        d_loss = adversarial_discriminator_loss(
            scores_real_p + scores_real_s,
            scores_fake_p + scores_fake_s,
        )
        d_loss.backward()
        self.opt_d.step()

        # === Step 2: 训练生成器 (Codec) ===
        self.opt_g.zero_grad()

        time_loss = self.wav_loss(x_recon_t, x_real_t)
        sc_loss, mag_loss = self.mr_stft_loss(x_recon_t, x_real_t)
        mel_loss = self.mel_loss(x_recon_t, x_real_t)

        scores_fake_p, feats_fake_p = self.mpd(x_recon_t)
        scores_fake_s, feats_fake_s = self.msd(x_recon_t)
        _, feats_real_p = self.mpd(x_real_t)
        _, feats_real_s = self.msd(x_real_t)

        adv_loss = adversarial_generator_loss(scores_fake_p + scores_fake_s)
        feat_loss = feature_matching_loss(
            feats_real_p + feats_real_s,
            feats_fake_p + feats_fake_s,
        )

        g_loss = (
            self.lambda_time * time_loss
            + self.lambda_freq * (sc_loss + mag_loss)
            + self.lambda_mel * mel_loss
            + self.lambda_adv * adv_loss
            + self.lambda_feat * feat_loss
            + self.lambda_commit * vq_losses['commitment_loss']
        )
        g_loss.backward()
        self.opt_g.step()

        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'time_loss': time_loss.item(),
            'stft_sc': sc_loss.item(),
            'stft_mag': mag_loss.item(),
            'mel_loss': mel_loss.item(),
            'adv_loss': adv_loss.item(),
            'feat_loss': feat_loss.item(),
            'commit_loss': vq_losses['commitment_loss'].item(),
            'codebook_util': vq_losses['avg_codebook_utilization'],
        }


# =============================================================================
# 主程序：完整演示
# =============================================================================

def generate_test_audio(
    batch_size: int, length: int, sample_rate: int = 24000
) -> torch.Tensor:
    """生成测试用的合成音频"""
    t = torch.linspace(0, length / sample_rate, length)
    batch = []
    for _ in range(batch_size):
        wave = torch.zeros(length)
        for _ in range(torch.randint(2, 6, (1,)).item()):
            freq = torch.rand(1).item() * 4000 + 100
            amp = torch.rand(1).item() * 0.3 + 0.1
            phase = torch.rand(1).item() * 2 * math.pi
            wave += amp * torch.sin(2 * math.pi * freq * t + phase)
        wave = wave / (wave.abs().max() + 1e-8) * 0.9
        batch.append(wave)
    return torch.stack(batch).unsqueeze(1)


if __name__ == "__main__":
    print("=" * 70)
    print("Stage 8: 完整的 Neural Audio Codec")
    print("=" * 70)

    sample_rate = 16000
    codec = NeuralAudioCodec(
        sample_rate=sample_rate,
        base_channels=16,        # 缩小模型便于演示
        code_dim=64,
        encoder_strides=(2, 4, 5, 8),
        num_quantizers=8,
        num_codes=1024,
    )

    total_params = sum(p.numel() for p in codec.parameters())
    print(f"\n模型参数量: {total_params:,}")
    print(f"采样率: {sample_rate} Hz")
    print(f"Hop length: {codec.hop_length}")
    print(f"帧率: {sample_rate / codec.hop_length:.1f} fps")

    print("\n--- 比特率 ---")
    for n_q in [1, 2, 4, 8]:
        br = codec.bitrate(n_q)
        print(f"  {n_q} 层 RVQ → {br/1000:.1f} kbps")

    print("\n--- 1. 编码-解码测试 ---")
    x = generate_test_audio(2, sample_rate, sample_rate)  # 1秒音频
    print(f"  输入: {list(x.shape)}")

    x_recon, codes, vq_losses = codec(x)
    print(f"  重建: {list(x_recon.shape)}")
    print(f"  码本索引: {list(codes.shape)}  (batch, quantizers, frames)")
    print(f"  VQ 损失: {vq_losses['commitment_loss'].item():.4f}")

    print("\n--- 2. 压缩/解压缩 ---")
    compressed = codec.compress(x)
    x_decoded = codec.decompress(compressed)
    print(f"  压缩前: {list(x.shape)} → 码: {list(compressed['codes'].shape)}")
    print(f"  解压后: {list(x_decoded.shape)}")

    original_bits = x.numel() * 32
    compressed_bits = compressed['codes'].numel() * math.log2(1024)
    print(f"  原始大小: {original_bits / 8 / 1024:.1f} KB")
    print(f"  压缩大小: {compressed_bits / 8 / 1024:.1f} KB")
    print(f"  压缩比:   {original_bits / compressed_bits:.1f}x")

    print("\n--- 3. 训练演示 (5步) ---")
    trainer = CodecTrainer(
        codec, sample_rate=sample_rate,
        lr_g=1e-4, lr_d=1e-4,
    )

    for step in range(5):
        batch = generate_test_audio(4, sample_rate, sample_rate)
        losses = trainer.train_step(batch)

        if (step + 1) % 1 == 0:
            print(f"\n  Step {step+1}:")
            print(f"    G loss: {losses['g_loss']:.4f}  |  D loss: {losses['d_loss']:.4f}")
            print(f"    Time: {losses['time_loss']:.4f}  |  "
                  f"STFT: {losses['stft_sc']:.4f}/{losses['stft_mag']:.4f}  |  "
                  f"Mel: {losses['mel_loss']:.4f}")
            print(f"    Adv: {losses['adv_loss']:.4f}  |  "
                  f"FM: {losses['feat_loss']:.4f}  |  "
                  f"Commit: {losses['commit_loss']:.4f}  |  "
                  f"CB利用率: {losses['codebook_util']:.1%}")

    print("\n--- 4. 可变比特率推理 ---")
    x_test = generate_test_audio(1, sample_rate, sample_rate)
    print("  不同量化层数下的重建质量:")
    with torch.no_grad():
        for n_q in [1, 2, 4, 8]:
            x_r, _, _ = codec(x_test, num_quantizers=n_q)
            min_len = min(x_test.shape[-1], x_r.shape[-1])
            mse = F.mse_loss(x_r[..., :min_len], x_test[..., :min_len]).item()
            br = codec.bitrate(n_q)
            print(f"    {n_q}层 ({br/1000:.1f} kbps): MSE = {mse:.4f}")

    print("\n" + "=" * 70)
    print("恭喜！你已经构建了一个完整的 Neural Audio Codec！")
    print("=" * 70)
    print("""
回顾学习路径:

  Stage 1: 音频基础
    └─ 数字音频、STFT、梅尔频谱 → 理解音频的数学表示

  Stage 2: 自编码器
    └─ 编码-瓶颈-解码结构 → 理解信息压缩的核心思想

  Stage 3: 向量量化 (VQ)
    └─ 连续→离散, STE, 码本 → 实现可传输的整数编码

  Stage 4: 残差向量量化 (RVQ)
    └─ 多层残差量化 → 高精度 + 可变比特率

  Stage 5: 卷积编解码器 (SEANet)
    └─ 因果卷积、膨胀残差块、多级下采样 → 专业级音频处理架构

  Stage 6: 判别器
    └─ MPD + MSD + Feature Matching → 对抗训练提升音质

  Stage 7: 感知损失
    └─ 多分辨率STFT + 梅尔损失 → 多角度衡量重建质量

  Stage 8: 完整整合 ← 你在这里!
    └─ 所有组件组合为端到端系统

进阶方向:
  - 添加带宽条件 (Bandwidth conditioning) 实现单模型多比特率
  - 实现流式推理 (Streaming inference) 用于实时通信
  - 添加语义 token (Semantic tokens) 用于语音合成 (VALL-E, MusicGen)
  - 探索 DAC 的改进: 改进的判别器、改进的量化器丢弃策略
  - 在真实数据集 (LibriSpeech, MUSDB) 上训练并评估 PESQ/ViSQOL 指标
""")
