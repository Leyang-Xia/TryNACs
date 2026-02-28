"""
Stage 2: 音频自编码器 (Autoencoder)
====================================

学习目标:
  1. 理解自编码器的"编码-瓶颈-解码"结构
  2. 用最简单的全连接网络实现音频自编码器
  3. 理解"信息瓶颈"如何迫使网络学习压缩表示
  4. 观察瓶颈维度对重建质量的影响

与 Neural Audio Codec 的关系:
  Neural Audio Codec (如 EnCodec, SoundStream) 的核心就是一个自编码器，
  只不过使用了更复杂的架构（卷积网络）和离散化瓶颈（向量量化）。
  这一阶段先用最简单的连续瓶颈理解核心思想。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAudioAutoencoder(nn.Module):
    """
    最简单的音频自编码器：全连接网络。

    结构:
        输入波形片段 (frame_size,)
          → Encoder: 线性层逐步降维
            → 瓶颈 (bottleneck_dim,)  ← 压缩后的表示
          → Decoder: 线性层逐步升维
        → 重建波形 (frame_size,)

    信息瓶颈原理:
        frame_size >> bottleneck_dim，网络被迫只保留最重要的信息。
        bottleneck_dim 越小 → 压缩率越高 → 重建质量越低（信息损失越多）
    """

    def __init__(self, frame_size: int = 256, bottleneck_dim: int = 32):
        super().__init__()
        self.frame_size = frame_size
        self.bottleneck_dim = bottleneck_dim

        self.encoder = nn.Sequential(
            nn.Linear(frame_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, bottleneck_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, frame_size),
            nn.Tanh(),  # 音频范围 [-1, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码：将输入帧压缩为低维表示"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码：从低维表示重建波形帧"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, frame_size) 输入波形帧
        Returns:
            reconstructed: (B, frame_size) 重建的波形帧
            z: (B, bottleneck_dim) 瓶颈表示
        """
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, z

    def compression_ratio(self) -> float:
        """计算压缩比"""
        return self.frame_size / self.bottleneck_dim


# =============================================================================
# 1D 卷积自编码器 — 更接近真实 Audio Codec 的架构
# =============================================================================

class Conv1dAutoencoder(nn.Module):
    """
    一维卷积自编码器。

    相比全连接网络的优势：
      1. 参数共享：卷积核在时间轴上滑动，大大减少参数量
      2. 局部感知：捕捉局部时间模式（语音的音素、音乐的节拍）
      3. 平移不变性：同一模式在不同时间位置都能被识别

    使用 stride > 1 的卷积来实现下采样（类似于编码/压缩），
    使用转置卷积来实现上采样（类似于解码/重建）。
    这与 SoundStream/EnCodec 的核心思想一致。
    """

    def __init__(self, channels: int = 1, bottleneck_channels: int = 8):
        super().__init__()

        self.encoder = nn.Sequential(
            # (B, 1, T) -> (B, 32, T/2)
            nn.Conv1d(channels, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            # (B, 32, T/2) -> (B, 64, T/4)
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            # (B, 64, T/4) -> (B, bottleneck_channels, T/4)
            nn.Conv1d(64, bottleneck_channels, kernel_size=3, stride=1, padding=1),
        )

        self.decoder = nn.Sequential(
            # (B, bottleneck_channels, T/4) -> (B, 64, T/4)
            nn.Conv1d(bottleneck_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # (B, 64, T/4) -> (B, 32, T/2)
            nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.ReLU(),
            # (B, 32, T/2) -> (B, 1, T)
            nn.ConvTranspose1d(32, channels, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 1, T) 波形
        Returns:
            reconstructed: (B, 1, T) 重建波形
            z: (B, C, T') 瓶颈表示，T' = T/4
        """
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, z


# =============================================================================
# 训练循环
# =============================================================================

def generate_training_data(
    num_samples: int = 1000,
    frame_size: int = 256,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """生成随机频率的正弦波作为训练数据"""
    data = []
    t = torch.linspace(0, frame_size / sample_rate, frame_size)
    for _ in range(num_samples):
        freq = torch.rand(1).item() * 2000 + 100  # 100-2100 Hz
        phase = torch.rand(1).item() * 2 * torch.pi
        amplitude = torch.rand(1).item() * 0.8 + 0.1
        wave = amplitude * torch.sin(2 * torch.pi * freq * t + phase)
        data.append(wave)
    return torch.stack(data)


def train_simple_autoencoder(
    bottleneck_dim: int = 32,
    num_epochs: int = 50,
    frame_size: int = 256,
) -> SimpleAudioAutoencoder:
    """训练全连接自编码器"""
    model = SimpleAudioAutoencoder(frame_size=frame_size, bottleneck_dim=bottleneck_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = generate_training_data(num_samples=2000, frame_size=frame_size)

    model.train()
    for epoch in range(num_epochs):
        idx = torch.randperm(data.size(0))[:256]
        batch = data[idx]
        reconstructed, z = model(batch)
        loss = F.mse_loss(reconstructed, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | Loss: {loss.item():.6f} | "
                  f"压缩比: {model.compression_ratio():.1f}x")

    return model


def train_conv_autoencoder(
    bottleneck_channels: int = 8,
    num_epochs: int = 50,
    segment_length: int = 1024,
) -> Conv1dAutoencoder:
    """训练卷积自编码器"""
    model = Conv1dAutoencoder(channels=1, bottleneck_channels=bottleneck_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(num_epochs):
        batch = generate_training_data(num_samples=64, frame_size=segment_length)
        batch = batch.unsqueeze(1)  # (B, 1, T)
        reconstructed, z = model(batch)
        loss = F.mse_loss(reconstructed, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            time_compression = batch.shape[-1] / z.shape[-1]
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | Loss: {loss.item():.6f} | "
                  f"时间压缩: {time_compression:.0f}x | "
                  f"瓶颈shape: {list(z.shape)}")

    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 2: 音频自编码器")
    print("=" * 60)

    print("\n--- 1. 全连接自编码器 ---")
    print("\n[瓶颈维度=32, 压缩比=8x]")
    model_32 = train_simple_autoencoder(bottleneck_dim=32)

    print("\n[瓶颈维度=8, 压缩比=32x]")
    model_8 = train_simple_autoencoder(bottleneck_dim=8)

    print("\n→ 观察：瓶颈越小，loss 越高（重建越困难）")

    print("\n--- 2. 测试重建质量 ---")
    test_data = generate_training_data(num_samples=100, frame_size=256)
    with torch.no_grad():
        recon_32, _ = model_32(test_data)
        recon_8, _ = model_8(test_data)
        mse_32 = F.mse_loss(recon_32, test_data).item()
        mse_8 = F.mse_loss(recon_8, test_data).item()
    print(f"  瓶颈=32 测试 MSE: {mse_32:.6f}")
    print(f"  瓶颈=8  测试 MSE: {mse_8:.6f}")

    print("\n--- 3. 1D 卷积自编码器 ---")
    print("\n这更接近真实 Audio Codec 的架构：")
    conv_model = train_conv_autoencoder(bottleneck_channels=8, num_epochs=50)

    print("\n--- 4. 模型参数量对比 ---")
    fc_params = sum(p.numel() for p in model_32.parameters())
    conv_params = sum(p.numel() for p in conv_model.parameters())
    print(f"  全连接自编码器参数量: {fc_params:,}")
    print(f"  卷积自编码器参数量:   {conv_params:,}")
    print(f"  卷积网络参数更少，但能处理任意长度输入！")

    print("\n✓ Stage 2 完成！")
    print("  关键理解：自编码器通过信息瓶颈实现压缩，瓶颈越窄压缩率越高。")
    print("  但目前瓶颈是连续值 → 无法直接用整数码本传输。")
    print("  下一步 → Stage 3: 向量量化 (VQ) — 将连续瓶颈离散化")
