"""
Stage 5: 卷积编解码器 (Convolutional Encoder-Decoder)
=====================================================

学习目标:
  1. 理解 EnCodec/SoundStream 所用的 SEANet 风格编解码器架构
  2. 掌握因果卷积 (Causal Convolution) — 支持流式推理
  3. 理解多尺度下采样/上采样的设计
  4. 实现残差块 (Residual Block) 和带权重归一化的卷积

架构核心思想:
  编码器: 原始波形 → 多级下采样 → 低维特征序列
  解码器: 低维特征 → 多级上采样 → 重建波形

  下采样比例由 strides 参数决定，例如 [2, 4, 5, 8] 意味着
  总下采样率 = 2 × 4 × 5 × 8 = 320
  即每 320 个音频采样点 → 1 个特征向量

  这与 EnCodec 的设计一致：24kHz 采样率时，
  320 采样 = 13.3ms ≈ 75 帧/秒
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence


# =============================================================================
# 基础构建块
# =============================================================================

class CausalConv1d(nn.Module):
    """
    因果卷积：只使用当前和过去的输入，不使用未来的输入。

    普通卷积:  padding = (kernel_size - 1) // 2  → 两边填充，看到未来
    因果卷积:  padding = kernel_size - 1          → 只在左边填充，截断右边

    为什么需要因果卷积？
      流式(streaming)推理时，我们不能等待未来的音频帧到达。
      因果卷积确保每个输出只依赖于已接收的输入。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = True,
    ):
        super().__init__()
        self.causal = causal
        self.stride = stride
        padding = (kernel_size - 1) * dilation
        self.causal_padding = padding if causal else padding // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation, groups=groups,
            bias=bias, padding=0 if causal else padding // 2,
        )
        nn.utils.parametrizations.weight_norm(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            x = F.pad(x, (self.causal_padding, 0))
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    """
    因果转置卷积（上采样）。

    转置卷积会在输出中引入额外的 padding，
    因果版本需要截断右侧多余的样本以保持因果性。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = True,
    ):
        super().__init__()
        self.causal = causal
        self.stride = stride
        self.conv_t = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0,
        )
        nn.utils.parametrizations.weight_norm(self.conv_t)
        self.trim = kernel_size - stride if causal else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv_t(x)
        if self.trim > 0:
            y = y[..., :-self.trim]
        return y


class ResidualBlock(nn.Module):
    """
    残差块：使用膨胀卷积捕捉不同尺度的时间模式。

    膨胀卷积 (Dilated Convolution):
      dilation=1: 看 3 个连续样本  (局部模式)
      dilation=3: 看 3 个间隔为3的样本 (中等范围模式)
      dilation=9: 看 3 个间隔为9的样本 (长程模式)

    组合使用不同膨胀率，让网络同时捕捉局部和全局特征。
    残差连接让梯度更容易传播，支持更深的网络。
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        causal: bool = True,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.ELU(),
            CausalConv1d(channels, channels, kernel_size, dilation=dilation, causal=causal),
            nn.ELU(),
            CausalConv1d(channels, channels, kernel_size=1, causal=causal),
        )
        self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.block(x)


class EncoderBlock(nn.Module):
    """
    编码器块：残差块序列 + 下采样。

    每个编码器块：
      1. 多个膨胀残差块（提取特征）
      2. 步进卷积（下采样，降低时间分辨率）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        num_residual: int = 3,
        dilation_base: int = 3,
        causal: bool = True,
    ):
        super().__init__()
        layers = []
        for i in range(num_residual):
            dilation = dilation_base ** i
            layers.append(ResidualBlock(in_channels, dilation=dilation, causal=causal))
        layers.append(nn.ELU())
        layers.append(CausalConv1d(
            in_channels, out_channels,
            kernel_size=2 * stride, stride=stride, causal=causal,
        ))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    解码器块：上采样 + 残差块序列。

    与编码器块对称：
      1. 转置卷积（上采样，提高时间分辨率）
      2. 多个膨胀残差块（细化特征）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        num_residual: int = 3,
        dilation_base: int = 3,
        causal: bool = True,
    ):
        super().__init__()
        layers = [
            nn.ELU(),
            CausalConvTranspose1d(
                in_channels, out_channels,
                kernel_size=2 * stride, stride=stride, causal=causal,
            ),
        ]
        for i in range(num_residual):
            dilation = dilation_base ** i
            layers.append(ResidualBlock(out_channels, dilation=dilation, causal=causal))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# =============================================================================
# 完整的编码器和解码器
# =============================================================================

class SEANetEncoder(nn.Module):
    """
    SEANet 风格编码器 (用于 EnCodec)。

    结构:
      输入 (B, 1, T)
      → 初始卷积 (B, C, T)
      → EncoderBlock_1 (B, 2C, T/s1)
      → EncoderBlock_2 (B, 4C, T/(s1*s2))
      → EncoderBlock_3 (B, 8C, T/(s1*s2*s3))
      → EncoderBlock_4 (B, 16C, T/(s1*s2*s3*s4))
      → 最终卷积 (B, D, T')
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        code_dim: int = 128,
        strides: Sequence[int] = (2, 4, 5, 8),
        num_residual: int = 3,
        causal: bool = True,
    ):
        super().__init__()
        self.strides = strides
        self.hop_length = math.prod(strides)

        layers = [CausalConv1d(in_channels, base_channels, kernel_size=7, causal=causal)]
        ch = base_channels
        for stride in strides:
            layers.append(EncoderBlock(ch, ch * 2, stride=stride,
                                       num_residual=num_residual, causal=causal))
            ch *= 2
        layers.append(nn.ELU())
        layers.append(CausalConv1d(ch, code_dim, kernel_size=3, causal=causal))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, T) 波形
        Returns:
            z: (B, code_dim, T') 其中 T' ≈ T / hop_length
        """
        return self.model(x)


class SEANetDecoder(nn.Module):
    """
    SEANet 风格解码器 — 编码器的镜像结构。

    结构（与编码器对称）:
      输入 (B, D, T')
      → 初始卷积 (B, 16C, T')
      → DecoderBlock_1 (B, 8C, T'*s4)
      → DecoderBlock_2 (B, 4C, T'*s4*s3)
      → DecoderBlock_3 (B, 2C, T'*s4*s3*s2)
      → DecoderBlock_4 (B, C, T'*s4*s3*s2*s1)
      → 最终卷积 (B, 1, T)
    """

    def __init__(
        self,
        out_channels: int = 1,
        base_channels: int = 32,
        code_dim: int = 128,
        strides: Sequence[int] = (2, 4, 5, 8),
        num_residual: int = 3,
        causal: bool = True,
    ):
        super().__init__()
        ch = base_channels * (2 ** len(strides))

        layers = [CausalConv1d(code_dim, ch, kernel_size=7, causal=causal)]
        for stride in reversed(strides):
            layers.append(DecoderBlock(ch, ch // 2, stride=stride,
                                       num_residual=num_residual, causal=causal))
            ch //= 2
        layers.append(nn.ELU())
        layers.append(CausalConv1d(ch, out_channels, kernel_size=7, causal=causal))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, code_dim, T')
        Returns:
            x: (B, 1, T) 重建波形
        """
        return self.model(z)


# =============================================================================
# 测试
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 5: 卷积编解码器 (SEANet 风格)")
    print("=" * 60)

    print("\n--- 1. 因果卷积 vs 普通卷积 ---")
    causal_conv = CausalConv1d(1, 16, kernel_size=7, causal=True)
    normal_conv = CausalConv1d(1, 16, kernel_size=7, causal=False)
    x = torch.randn(1, 1, 100)
    y_causal = causal_conv(x)
    y_normal = normal_conv(x)
    print(f"  输入: {list(x.shape)}")
    print(f"  因果卷积输出: {list(y_causal.shape)}  (保持长度，只看过去)")
    print(f"  普通卷积输出: {list(y_normal.shape)}  (保持长度，看两侧)")

    print("\n--- 2. 残差块 ---")
    res = ResidualBlock(channels=64, dilation=3)
    x = torch.randn(2, 64, 100)
    y = res(x)
    print(f"  输入: {list(x.shape)} → 输出: {list(y.shape)} (shape 不变)")

    print("\n--- 3. 编码器 ---")
    strides = (2, 4, 5, 8)
    hop = math.prod(strides)
    encoder = SEANetEncoder(strides=strides, code_dim=128)
    x = torch.randn(2, 1, 16000)  # 1秒 16kHz
    z = encoder(x)
    print(f"  输入波形:    {list(x.shape)}")
    print(f"  编码输出:    {list(z.shape)}")
    print(f"  总下采样率:  {hop}x")
    print(f"  时间压缩:    {x.shape[-1]} → {z.shape[-1]} 帧")
    print(f"  编码器参数:  {count_parameters(encoder):,}")

    print("\n--- 4. 解码器 ---")
    decoder = SEANetDecoder(strides=strides, code_dim=128)
    x_recon = decoder(z)
    print(f"  解码输入:    {list(z.shape)}")
    print(f"  重建波形:    {list(x_recon.shape)}")
    print(f"  解码器参数:  {count_parameters(decoder):,}")

    print("\n--- 5. 编码-解码往返 ---")
    length_match = x_recon.shape[-1] == x.shape[-1]
    print(f"  输入输出长度匹配: {length_match}")
    if not length_match:
        print(f"  长度差异: {x.shape[-1] - x_recon.shape[-1]} (可通过 padding 解决)")

    print("\n--- 6. 不同下采样配置 ---")
    configs = [
        ((2, 4, 5, 8), "EnCodec 默认"),
        ((2, 4, 8, 8), "SoundStream"),
        ((4, 5, 6, 8), "高压缩率"),
    ]
    for strides, name in configs:
        hop = math.prod(strides)
        enc = SEANetEncoder(strides=strides, code_dim=128)
        z = enc(torch.randn(1, 1, 24000))
        print(f"  {name}: strides={strides}, hop={hop}, "
              f"24kHz→{24000//hop}帧/秒, 参数={count_parameters(enc):,}")

    print("\n✓ Stage 5 完成！")
    print("  关键理解：SEANet 编解码器通过多级步进卷积实现时间压缩，")
    print("  膨胀残差块捕捉多尺度时间模式，因果卷积支持流式推理。")
    print("  下一步 → Stage 6: 判别器 — 对抗训练提升音质")
