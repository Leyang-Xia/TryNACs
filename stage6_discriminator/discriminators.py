"""
Stage 6: 判别器与对抗训练 (Discriminators & Adversarial Training)
================================================================

学习目标:
  1. 理解为什么单纯的 MSE/L1 损失不够（过度平滑问题）
  2. 实现多尺度判别器 (Multi-Scale Discriminator, MSD)
  3. 实现多周期判别器 (Multi-Period Discriminator, MPD)
  4. 理解 Feature Matching Loss 如何稳定对抗训练

为什么需要对抗训练？
  仅用 MSE 损失训练解码器，重建音频往往听起来"模糊"、缺乏细节。
  这是因为 MSE 鼓励模型输出所有可能输出的"平均值"。

  GAN 判别器迫使生成器（解码器）产生在统计上更接近真实音频分布的输出。
  → 更锐利的频谱细节、更自然的音质

  SoundStream 使用 MSD，HiFi-GAN / EnCodec 同时使用 MSD + MPD。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 多周期判别器 (Multi-Period Discriminator)
# =============================================================================

class PeriodDiscriminator(nn.Module):
    """
    单周期子判别器。

    核心思想：将1D音频按照特定周期 p 重塑为2D，
    然后用2D卷积分析周期性模式。

    例如 period=2:
      [s0, s1, s2, s3, s4, s5, ...] → [[s0, s1],
                                         [s2, s3],
                                         [s4, s5], ...]

    不同周期捕捉不同的周期性结构：
      - period=2: 奇偶样本关系
      - period=3: 三拍节奏
      - period=5: 基频相关模式
    """

    def __init__(self, period: int, channels: int = 32, max_channels: int = 512):
        super().__init__()
        self.period = period

        layers = []
        in_ch = 1
        for i in range(4):
            out_ch = min(channels * (2 ** (i + 1)), max_channels)
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)))
            layers.append(nn.LeakyReLU(0.1))
            in_ch = out_ch

        layers.append(nn.Conv2d(in_ch, in_ch, kernel_size=(5, 1), padding=(2, 0)))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.Conv2d(in_ch, 1, kernel_size=(3, 1), padding=(1, 0)))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: (B, 1, T) 波形
        Returns:
            score: 判别得分
            features: 中间层特征（用于 feature matching loss）
        """
        B, C, T = x.shape
        # 按周期重塑为 2D
        if T % self.period != 0:
            pad = self.period - (T % self.period)
            x = F.pad(x, (0, pad), mode='reflect')
            T = T + pad
        x = x.view(B, C, T // self.period, self.period)

        features = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                features.append(x)
        return x, features


class MultiPeriodDiscriminator(nn.Module):
    """
    多周期判别器 (MPD) — HiFi-GAN 提出，EnCodec 沿用。

    使用多个不同周期的子判别器，每个关注不同的周期性结构。
    """

    def __init__(self, periods: list[int] = None):
        super().__init__()
        if periods is None:
            periods = [2, 3, 5, 7, 11]  # 互质数，确保覆盖不同周期
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """
        Returns:
            scores: 每个子判别器的得分
            features: 每个子判别器的中间特征
        """
        scores, features = [], []
        for d in self.discriminators:
            s, f = d(x)
            scores.append(s)
            features.append(f)
        return scores, features


# =============================================================================
# 多尺度判别器 (Multi-Scale Discriminator)
# =============================================================================

class ScaleDiscriminator(nn.Module):
    """
    单尺度子判别器：1D 卷积分析不同时间尺度的模式。
    使用分组卷积减少参数量。
    """

    def __init__(self, channels: int = 128):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(1, channels, kernel_size=15, stride=1, padding=7),
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels, channels, kernel_size=41, stride=4, padding=20, groups=4),
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels, channels * 2, kernel_size=41, stride=4, padding=20, groups=16),
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels * 2, channels * 4, kernel_size=41, stride=4, padding=20, groups=16),
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels * 4, channels * 4, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels * 4, 1, kernel_size=3, stride=1, padding=1),
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                features.append(x)
        return x, features


class MultiScaleDiscriminator(nn.Module):
    """
    多尺度判别器 (MSD) — SoundStream / EnCodec 使用。

    通过 AvgPool 逐步降采样输入音频，在不同时间分辨率上分析：
      - 原始分辨率: 捕捉高频细节
      - 2x 下采样:  捕捉中频结构
      - 4x 下采样:  捕捉低频包络
    """

    def __init__(self, num_scales: int = 3):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator() for _ in range(num_scales)
        ])
        self.downsamplers = nn.ModuleList([
            nn.Identity(),  # 原始分辨率
            *[nn.AvgPool1d(kernel_size=4, stride=2, padding=2)
              for _ in range(num_scales - 1)]
        ])

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        scores, features = [], []
        for disc, downsample in zip(self.discriminators, self.downsamplers):
            x_ds = downsample(x)
            s, f = disc(x_ds)
            scores.append(s)
            features.append(f)
        return scores, features


# =============================================================================
# 对抗训练损失函数
# =============================================================================

def adversarial_generator_loss(disc_fake_scores: list[torch.Tensor]) -> torch.Tensor:
    """
    生成器对抗损失（hinge 形式）。
    目标：让判别器误认为生成的音频是真实的。
    """
    loss = 0.0
    for score in disc_fake_scores:
        loss += torch.mean(F.relu(1 - score))
    return loss / len(disc_fake_scores)


def adversarial_discriminator_loss(
    disc_real_scores: list[torch.Tensor],
    disc_fake_scores: list[torch.Tensor],
) -> torch.Tensor:
    """
    判别器对抗损失（hinge 形式）。
    目标：正确区分真实和生成的音频。
    """
    loss = 0.0
    for real_score, fake_score in zip(disc_real_scores, disc_fake_scores):
        loss += torch.mean(F.relu(1 - real_score)) + torch.mean(F.relu(1 + fake_score))
    return loss / len(disc_real_scores)


def feature_matching_loss(
    real_features: list[list[torch.Tensor]],
    fake_features: list[list[torch.Tensor]],
) -> torch.Tensor:
    """
    特征匹配损失 (Feature Matching Loss)。

    不直接比较波形，而是比较判别器中间层的特征激活。
    迫使生成器在判别器看来的多个抽象层次上匹配真实音频的特征。

    这比纯对抗损失更稳定，因为它提供了更平滑的梯度信号。
    """
    loss = 0.0
    count = 0
    for real_feat_list, fake_feat_list in zip(real_features, fake_features):
        for real_feat, fake_feat in zip(real_feat_list, fake_feat_list):
            loss += F.l1_loss(fake_feat, real_feat.detach())
            count += 1
    return loss / max(count, 1)


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 6: 判别器与对抗训练")
    print("=" * 60)

    batch = 2
    x_real = torch.randn(batch, 1, 16000)
    x_fake = torch.randn(batch, 1, 16000)

    print("\n--- 1. 多周期判别器 (MPD) ---")
    mpd = MultiPeriodDiscriminator()
    scores_real, feats_real = mpd(x_real)
    scores_fake, feats_fake = mpd(x_fake)
    print(f"  子判别器数量: {len(mpd.discriminators)}")
    print(f"  得分 shapes: {[list(s.shape) for s in scores_real]}")
    print(f"  每个子判别器的特征层数: {[len(f) for f in feats_real]}")

    print("\n--- 2. 多尺度判别器 (MSD) ---")
    msd = MultiScaleDiscriminator(num_scales=3)
    scores_real_s, feats_real_s = msd(x_real)
    scores_fake_s, feats_fake_s = msd(x_fake)
    print(f"  子判别器数量: {len(msd.discriminators)}")
    print(f"  得分 shapes: {[list(s.shape) for s in scores_real_s]}")

    print("\n--- 3. 损失函数 ---")
    g_loss = adversarial_generator_loss(scores_fake + scores_fake_s)
    d_loss = adversarial_discriminator_loss(
        scores_real + scores_real_s,
        scores_fake + scores_fake_s,
    )
    fm_loss = feature_matching_loss(
        feats_real + feats_real_s,
        feats_fake + feats_fake_s,
    )
    print(f"  生成器对抗损失:    {g_loss.item():.4f}")
    print(f"  判别器对抗损失:    {d_loss.item():.4f}")
    print(f"  特征匹配损失:      {fm_loss.item():.4f}")

    print("\n--- 4. 模型参数量 ---")
    mpd_params = sum(p.numel() for p in mpd.parameters())
    msd_params = sum(p.numel() for p in msd.parameters())
    print(f"  MPD 参数: {mpd_params:,}")
    print(f"  MSD 参数: {msd_params:,}")
    print(f"  总判别器: {mpd_params + msd_params:,}")

    print("\n--- 5. 对抗训练模拟 (1步) ---")
    gen_param = nn.Parameter(torch.randn(batch, 1, 16000) * 0.01)
    opt_g = torch.optim.Adam([gen_param], lr=1e-3)
    opt_d = torch.optim.Adam(
        list(mpd.parameters()) + list(msd.parameters()), lr=1e-4
    )

    # 判别器步骤
    opt_d.zero_grad()
    with torch.no_grad():
        fake = gen_param.detach()
    scores_r_p, _ = mpd(x_real)
    scores_f_p, _ = mpd(fake)
    scores_r_s, _ = msd(x_real)
    scores_f_s, _ = msd(fake)
    d_loss = adversarial_discriminator_loss(
        scores_r_p + scores_r_s, scores_f_p + scores_f_s
    )
    d_loss.backward()
    opt_d.step()
    print(f"  D loss: {d_loss.item():.4f}")

    # 生成器步骤
    opt_g.zero_grad()
    scores_f_p, feats_f_p = mpd(gen_param)
    scores_f_s, feats_f_s = msd(gen_param)
    _, feats_r_p = mpd(x_real)
    _, feats_r_s = msd(x_real)
    g_adv = adversarial_generator_loss(scores_f_p + scores_f_s)
    g_fm = feature_matching_loss(feats_r_p + feats_r_s, feats_f_p + feats_f_s)
    g_total = g_adv + 2.0 * g_fm
    g_total.backward()
    opt_g.step()
    print(f"  G loss: {g_total.item():.4f} (adv={g_adv.item():.4f}, fm={g_fm.item():.4f})")

    print("\n✓ Stage 6 完成！")
    print("  关键理解：判别器从多个角度分析音频的真实性，")
    print("  Feature Matching 提供稳定的训练信号。")
    print("  下一步 → Stage 7: 感知损失函数")
