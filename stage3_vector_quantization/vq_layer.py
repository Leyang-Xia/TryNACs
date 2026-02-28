"""
Stage 3: 向量量化 (Vector Quantization)
========================================

学习目标:
  1. 理解向量量化的核心思想：用有限的码本向量逼近连续空间
  2. 实现 VQ 层，包括码本查找和梯度直通估计 (Straight-Through Estimator)
  3. 理解 commitment loss 和 codebook loss 的作用
  4. 构建 VQ-VAE (Vector Quantized Variational Autoencoder)

为什么需要向量量化？
  Stage 2 的自编码器瓶颈是连续值，无法直接用整数传输。
  VQ 将连续向量映射到最近的码本向量（离散化），
  这样我们就可以只传输码本索引（整数），实现真正的压缩。

  例如: 码本大小 1024 → 每个向量只需 10 bits (2^10 = 1024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VectorQuantizer(nn.Module):
    """
    向量量化层 — Neural Audio Codec 的核心组件。

    工作流程:
      1. 接收编码器输出 z_e (连续向量)
      2. 在码本中找到最近邻向量 z_q
      3. 用 z_q 替换 z_e（离散化）
      4. 使用 straight-through estimator 让梯度绕过不可导的量化步骤

    码本 (Codebook):
      一个可学习的查找表，shape = (num_codes, code_dim)
      每个码本向量代表一种"原型"音频模式。

    参考: "Neural Discrete Representation Learning" (van den Oord et al., 2017)
    """

    def __init__(
        self,
        num_codes: int = 1024,
        code_dim: int = 64,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_weight = commitment_weight

        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    def find_nearest_code(self, z_e: torch.Tensor) -> torch.Tensor:
        """
        找到每个输入向量在码本中的最近邻。

        距离计算展开:
          ||z_e - e_j||^2 = ||z_e||^2 + ||e_j||^2 - 2 * z_e · e_j

        Args:
            z_e: (N, D) 编码器输出向量
        Returns:
            indices: (N,) 最近邻码本索引
        """
        # ||z_e||^2, shape (N, 1)
        z_e_sq = (z_e ** 2).sum(dim=-1, keepdim=True)
        # ||e_j||^2, shape (1, K)
        codebook_sq = (self.codebook.weight ** 2).sum(dim=-1, keepdim=True).t()
        # z_e · e_j, shape (N, K)
        dot_product = z_e @ self.codebook.weight.t()
        # 距离矩阵 (N, K)
        distances = z_e_sq + codebook_sq - 2 * dot_product
        # 最近邻索引
        indices = distances.argmin(dim=-1)
        return indices

    def forward(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            z_e: (B, D, T) 编码器输出, D=code_dim, T=时间步

        Returns:
            z_q: (B, D, T) 量化后的向量
            indices: (B, T) 码本索引
            losses: dict 包含各项损失

        损失解释:
          codebook_loss:  推动码本向量靠近编码器输出 → sg[z_e] - e
          commitment_loss: 推动编码器输出靠近码本向量 → z_e - sg[e]
          (sg = stop gradient)
        """
        B, D, T = z_e.shape

        # (B, D, T) -> (B*T, D)
        z_e_flat = rearrange(z_e, 'b d t -> (b t) d')

        indices = self.find_nearest_code(z_e_flat)
        z_q_flat = self.codebook(indices)  # (B*T, D)

        # 损失计算
        codebook_loss = F.mse_loss(z_q_flat, z_e_flat.detach())
        commitment_loss = F.mse_loss(z_e_flat, z_q_flat.detach())
        vq_loss = codebook_loss + self.commitment_weight * commitment_loss

        # ★ Straight-Through Estimator (STE) ★
        # 前向：使用量化后的 z_q
        # 反向：梯度直接传给 z_e（绕过不可导的 argmin）
        z_q_flat = z_e_flat + (z_q_flat - z_e_flat).detach()

        z_q = rearrange(z_q_flat, '(b t) d -> b d t', b=B, t=T)
        indices = indices.view(B, T)

        # 码本利用率统计
        unique_codes = indices.unique().numel()
        utilization = unique_codes / self.num_codes

        losses = {
            'vq_loss': vq_loss,
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss,
            'codebook_utilization': utilization,
        }
        return z_q, indices, losses

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """根据索引从码本取回向量（解码端使用）"""
        z_q = self.codebook(indices)
        return rearrange(z_q, 'b t d -> b d t')

    def bits_per_sample(self, hop_length: int, sample_rate: int) -> float:
        """
        计算比特率。
        每 hop_length 个音频采样产生一个码本索引，
        每个索引需要 log2(num_codes) bits。
        """
        import math
        bits_per_code = math.log2(self.num_codes)
        codes_per_second = sample_rate / hop_length
        return bits_per_code * codes_per_second


# =============================================================================
# VQ-VAE: 将 VQ 层嵌入自编码器
# =============================================================================

class VQVAE(nn.Module):
    """
    VQ-VAE: 带向量量化瓶颈的自编码器。

    这是 Neural Audio Codec 的直接原型：
      Encoder → VQ → Decoder

    与 Stage 2 的自编码器的关键区别：
      瓶颈不再是连续值，而是离散的码本索引。
      这意味着我们可以用整数来表示音频！
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 64,
        num_codes: int = 1024,
        code_dim: int = 64,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, code_dim, kernel_size=3, padding=1),
        )

        self.vq = VectorQuantizer(
            num_codes=num_codes,
            code_dim=code_dim,
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(code_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, 32, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, in_channels, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """编码 + 量化"""
        z_e = self.encoder(x)
        z_q, indices, vq_losses = self.vq(z_e)
        return z_q, indices, vq_losses

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """从量化表示解码"""
        return self.decoder(z_q)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        z_q, indices, vq_losses = self.encode(x)
        x_recon = self.decode(z_q)
        return x_recon, indices, vq_losses

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """编码：波形 → 码本索引 (整数)"""
        z_e = self.encoder(x)
        _, indices, _ = self.vq(z_e)
        return indices

    def decompress(self, indices: torch.Tensor) -> torch.Tensor:
        """解码：码本索引 → 波形"""
        z_q = self.vq.lookup(indices)
        return self.decode(z_q)


# =============================================================================
# 训练
# =============================================================================

def generate_data(batch_size: int = 64, length: int = 1024) -> torch.Tensor:
    """生成训练数据：随机正弦波组合"""
    t = torch.linspace(0, length / 16000, length)
    batch = []
    for _ in range(batch_size):
        num_harmonics = torch.randint(1, 5, (1,)).item()
        wave = torch.zeros(length)
        for _ in range(num_harmonics):
            freq = torch.rand(1).item() * 3000 + 100
            amp = torch.rand(1).item()
            phase = torch.rand(1).item() * 2 * 3.14159
            wave += amp * torch.sin(2 * 3.14159 * freq * t + phase)
        wave = wave / (wave.abs().max() + 1e-8)
        batch.append(wave)
    return torch.stack(batch).unsqueeze(1)  # (B, 1, T)


def train_vqvae(num_epochs: int = 100) -> VQVAE:
    """训练 VQ-VAE"""
    model = VQVAE(num_codes=512, code_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    model.train()
    for epoch in range(num_epochs):
        batch = generate_data(batch_size=32, length=1024)
        x_recon, indices, vq_losses = model(batch)

        recon_loss = F.mse_loss(x_recon, batch)
        total_loss = recon_loss + vq_losses['vq_loss']

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Recon: {recon_loss.item():.4f} | "
                  f"VQ: {vq_losses['vq_loss'].item():.4f} | "
                  f"码本利用率: {vq_losses['codebook_utilization']:.1%}")

    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 3: 向量量化 (Vector Quantization)")
    print("=" * 60)

    print("\n--- 1. VQ 层基本操作 ---")
    vq = VectorQuantizer(num_codes=256, code_dim=32)
    z_e = torch.randn(2, 32, 10)  # (B, D, T)
    z_q, indices, losses = vq(z_e)
    print(f"  输入 z_e:  shape={list(z_e.shape)}")
    print(f"  输出 z_q:  shape={list(z_q.shape)}")
    print(f"  码本索引:   shape={list(indices.shape)}, 范围=[{indices.min()}, {indices.max()}]")
    print(f"  VQ 损失:   {losses['vq_loss'].item():.4f}")
    print(f"  码本利用率: {losses['codebook_utilization']:.1%}")

    print("\n--- 2. 码本查找（解码端）---")
    z_recovered = vq.lookup(indices)
    print(f"  索引 → 向量: shape={list(z_recovered.shape)}")
    error = (z_q - z_recovered).abs().max().item()
    print(f"  查找精度验证 (应为0): {error:.2e}")

    print("\n--- 3. 比特率计算 ---")
    vq_1024 = VectorQuantizer(num_codes=1024, code_dim=64)
    bps = vq_1024.bits_per_sample(hop_length=320, sample_rate=16000)
    print(f"  码本=1024, hop=320, sr=16kHz → {bps:.0f} bps = {bps/1000:.1f} kbps")

    print("\n--- 4. 训练 VQ-VAE ---")
    model = train_vqvae(num_epochs=100)

    print("\n--- 5. 压缩与解压缩测试 ---")
    test_data = generate_data(batch_size=4, length=1024)
    with torch.no_grad():
        codes = model.compress(test_data)
        recon = model.decompress(codes)
        mse = F.mse_loss(recon, test_data).item()

    import math
    original_bits = test_data.numel() * 32  # float32
    compressed_bits = codes.numel() * math.log2(512)
    ratio = original_bits / compressed_bits

    print(f"  原始数据:  {test_data.shape} ({original_bits:,} bits)")
    print(f"  压缩码:    {codes.shape} ({compressed_bits:,.0f} bits)")
    print(f"  压缩比:    {ratio:.1f}x")
    print(f"  重建 MSE:  {mse:.4f}")

    print("\n✓ Stage 3 完成！")
    print("  关键理解：VQ 将连续瓶颈离散化为码本索引，实现真正的压缩。")
    print("  但单层VQ的表达能力有限 → 码本要么太大要么质量不够。")
    print("  下一步 → Stage 4: 残差向量量化 (RVQ) — 多层递进量化")
