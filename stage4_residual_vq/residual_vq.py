"""
Stage 4: 残差向量量化 (Residual Vector Quantization, RVQ)
=========================================================

学习目标:
  1. 理解 RVQ 的递进量化思想
  2. 实现多层残差量化
  3. 理解可变比特率 (Variable Bitrate) 的原理
  4. 对比 单层VQ vs RVQ 的性能差异

为什么需要 RVQ？
  Stage 3 中单层 VQ 面临一个两难困境：
    - 码本太小 → 精度不够
    - 码本太大 → 码本利用率低 (codebook collapse)，训练不稳定

  RVQ 的巧妙解决方案：使用多个小码本逐层量化"残差"。

  例子 (类比：十进制近似一个数)：
    要表示 3.14159...
    第1层: 量化到 3         (残差 = 0.14159)
    第2层: 量化到 0.1       (残差 = 0.04159)
    第3层: 量化到 0.04      (残差 = 0.00159)
    ...
    每一层都在修正上一层的误差！

  SoundStream / EnCodec / DAC 全部使用 RVQ 作为核心量化策略。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VectorQuantizerEMA(nn.Module):
    """
    使用 EMA (Exponential Moving Average) 更新的向量量化层。

    相比 Stage 3 的基础 VQ，EMA 更新有以下优势：
      1. 无需额外的 codebook loss
      2. 码本更新更稳定
      3. 不依赖梯度下降来更新码本
    """

    def __init__(
        self,
        num_codes: int = 1024,
        code_dim: int = 128,
        decay: float = 0.99,
        eps: float = 1e-5,
        commitment_weight: float = 1.0,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.decay = decay
        self.eps = eps
        self.commitment_weight = commitment_weight

        embed = torch.randn(num_codes, code_dim)
        self.register_buffer('embedding', embed)
        self.register_buffer('cluster_size', torch.zeros(num_codes))
        self.register_buffer('embed_avg', embed.clone())
        self._initialized = False

    def _init_from_data(self, z_flat: torch.Tensor):
        """使用第一批数据初始化码本（而非随机初始化）"""
        if self._initialized:
            return
        n = min(z_flat.shape[0], self.num_codes)
        indices = torch.randperm(z_flat.shape[0])[:n]
        self.embedding[:n] = z_flat[indices].detach()
        self._initialized = True

    def find_nearest(self, z_flat: torch.Tensor) -> torch.Tensor:
        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            + self.embedding.pow(2).sum(1)
            - 2 * z_flat @ self.embedding.t()
        )
        return dist.argmin(dim=-1)

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            z_e: (B, D, T)
        Returns:
            z_q, indices, losses
        """
        B, D, T = z_e.shape
        z_flat = rearrange(z_e, 'b d t -> (b t) d')

        if self.training and not self._initialized:
            self._init_from_data(z_flat)

        indices = self.find_nearest(z_flat)
        z_q_flat = self.embedding[indices]

        # EMA 更新（仅训练时）
        if self.training:
            one_hot = F.one_hot(indices, self.num_codes).float()  # (N, K)
            sum_counts = one_hot.sum(0)  # (K,)
            sum_embeddings = one_hot.t() @ z_flat  # (K, D)

            self.cluster_size.mul_(self.decay).add_(sum_counts, alpha=1 - self.decay)
            self.embed_avg.mul_(self.decay).add_(sum_embeddings, alpha=1 - self.decay)

            # Laplace smoothing
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps)
                / (n + self.num_codes * self.eps) * n
            )
            self.embedding.copy_(self.embed_avg / cluster_size.unsqueeze(1))

        commitment_loss = self.commitment_weight * F.mse_loss(z_flat, z_q_flat.detach())

        # STE
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()

        z_q = rearrange(z_q_flat, '(b t) d -> b d t', b=B, t=T)
        indices = indices.view(B, T)

        utilization = indices.unique().numel() / self.num_codes

        return z_q, indices, {
            'commitment_loss': commitment_loss,
            'codebook_utilization': utilization,
        }

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        z_q = self.embedding[indices]
        return rearrange(z_q, 'b t d -> b d t')


class ResidualVectorQuantizer(nn.Module):
    """
    残差向量量化 (RVQ) — SoundStream/EnCodec 的核心。

    工作原理：
      r_0 = z_e                    (初始残差 = 编码器输出)
      z_q_1, idx_1 = VQ_1(r_0)     (第1层量化)
      r_1 = r_0 - z_q_1            (第1层残差)
      z_q_2, idx_2 = VQ_2(r_1)     (第2层量化残差)
      r_2 = r_1 - z_q_2            (第2层残差)
      ...
      最终: z_q = z_q_1 + z_q_2 + ... + z_q_N

    比特率控制：
      - 每层使用 log2(num_codes) bits
      - N 层 RVQ 的总比特 = N * log2(num_codes)
      - 推理时可以只用前 k < N 层 → 降低比特率（牺牲质量）
    """

    def __init__(
        self,
        num_quantizers: int = 8,
        num_codes: int = 1024,
        code_dim: int = 128,
        commitment_weight: float = 1.0,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.num_codes = num_codes
        self.code_dim = code_dim

        self.quantizers = nn.ModuleList([
            VectorQuantizerEMA(
                num_codes=num_codes,
                code_dim=code_dim,
                commitment_weight=commitment_weight,
            )
            for _ in range(num_quantizers)
        ])

    def forward(
        self,
        z_e: torch.Tensor,
        num_quantizers: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            z_e: (B, D, T) 编码器输出
            num_quantizers: 使用的量化层数（默认全部），可用于可变比特率

        Returns:
            z_q: (B, D, T) 量化后的向量（所有层的和）
            all_indices: (B, N, T) 所有层的码本索引
            losses: dict
        """
        n_q = num_quantizers or self.num_quantizers
        n_q = min(n_q, self.num_quantizers)

        residual = z_e
        z_q_total = torch.zeros_like(z_e)
        all_indices = []
        total_commitment = 0.0
        total_utilization = 0.0

        for i in range(n_q):
            z_q_i, indices_i, losses_i = self.quantizers[i](residual)
            residual = residual - z_q_i  # ★ 关键：下一层量化残差
            z_q_total = z_q_total + z_q_i
            all_indices.append(indices_i)
            total_commitment += losses_i['commitment_loss']
            total_utilization += losses_i['codebook_utilization']

        all_indices = torch.stack(all_indices, dim=1)  # (B, N, T)

        return z_q_total, all_indices, {
            'commitment_loss': total_commitment / n_q,
            'avg_codebook_utilization': total_utilization / n_q,
            'num_quantizers_used': n_q,
        }

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        从码本索引重建量化向量。

        Args:
            indices: (B, N, T) 多层码本索引
        Returns:
            z_q: (B, D, T)
        """
        B, N, T = indices.shape
        z_q = torch.zeros(B, self.code_dim, T, device=indices.device)
        for i in range(N):
            z_q += self.quantizers[i].lookup(indices[:, i])
        return z_q

    def bitrate(self, hop_length: int, sample_rate: int, num_quantizers: int | None = None) -> float:
        """计算比特率 (bps)"""
        n_q = num_quantizers or self.num_quantizers
        bits_per_code = math.log2(self.num_codes)
        codes_per_second = sample_rate / hop_length
        return n_q * bits_per_code * codes_per_second


# =============================================================================
# 演示与对比
# =============================================================================

def compare_vq_vs_rvq():
    """对比单层VQ和RVQ的量化精度"""
    torch.manual_seed(42)
    code_dim = 64
    z = torch.randn(4, code_dim, 50)  # 随机"编码器输出"

    print("对比: 单层 VQ (1024 codes) vs 8层 RVQ (每层128 codes)")
    print(f"  单层VQ 总码本: 1024,  bits/step: {math.log2(1024):.0f}")
    print(f"  RVQ 总码本: 8×128=1024, bits/step: {8 * math.log2(128):.0f}")
    print()

    single_vq = VectorQuantizerEMA(num_codes=1024, code_dim=code_dim)
    single_vq._init_from_data(rearrange(z, 'b d t -> (b t) d'))
    z_q_single, _, _ = single_vq(z)
    mse_single = F.mse_loss(z_q_single, z).item()

    rvq = ResidualVectorQuantizer(num_quantizers=8, num_codes=128, code_dim=code_dim)
    for q in rvq.quantizers:
        q._init_from_data(rearrange(z, 'b d t -> (b t) d'))
    z_q_rvq, indices, losses = rvq(z)
    mse_rvq = F.mse_loss(z_q_rvq, z).item()

    print(f"  单层VQ MSE: {mse_single:.4f}")
    print(f"  RVQ MSE:    {mse_rvq:.4f}")
    print(f"  RVQ 优势: {mse_single / max(mse_rvq, 1e-8):.1f}x 更精确")


def demo_variable_bitrate():
    """演示可变比特率：使用不同层数的 RVQ"""
    torch.manual_seed(0)
    code_dim = 64
    z = torch.randn(4, code_dim, 50)

    rvq = ResidualVectorQuantizer(num_quantizers=8, num_codes=1024, code_dim=code_dim)
    for q in rvq.quantizers:
        q._init_from_data(rearrange(z, 'b d t -> (b t) d'))

    print("可变比特率 — 使用不同数量的量化层:")
    print(f"  {'层数':>4} | {'比特率':>12} | {'量化 MSE':>10}")
    print("  " + "-" * 36)

    for n_q in [1, 2, 4, 6, 8]:
        z_q, _, _ = rvq(z, num_quantizers=n_q)
        mse = F.mse_loss(z_q, z).item()
        br = rvq.bitrate(hop_length=320, sample_rate=16000, num_quantizers=n_q)
        print(f"  {n_q:>4} | {br/1000:>8.1f} kbps | {mse:>10.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 4: 残差向量量化 (RVQ)")
    print("=" * 60)

    print("\n--- 1. RVQ 基本操作 ---")
    rvq = ResidualVectorQuantizer(num_quantizers=8, num_codes=1024, code_dim=128)
    z = torch.randn(2, 128, 20)
    z_q, indices, losses = rvq(z)
    print(f"  输入:     shape={list(z.shape)}")
    print(f"  量化输出: shape={list(z_q.shape)}")
    print(f"  索引:     shape={list(indices.shape)}  (B, num_quantizers, T)")
    print(f"  Commitment loss: {losses['commitment_loss']:.4f}")
    print(f"  平均码本利用率:  {losses['avg_codebook_utilization']:.1%}")

    print("\n--- 2. 从索引解码 ---")
    z_q_decoded = rvq.decode_from_indices(indices)
    print(f"  解码输出: shape={list(z_q_decoded.shape)}")
    error = (z_q - z_q_decoded).abs().max().item()
    print(f"  解码精度验证: {error:.2e}")

    print("\n--- 3. 单层VQ vs RVQ 对比 ---")
    compare_vq_vs_rvq()

    print("\n--- 4. 可变比特率 ---")
    demo_variable_bitrate()

    print("\n--- 5. 比特率计算 ---")
    for n_q in [2, 4, 8]:
        br = rvq.bitrate(hop_length=320, sample_rate=24000, num_quantizers=n_q)
        print(f"  {n_q}层 RVQ, 码本1024, hop=320, sr=24kHz → {br/1000:.1f} kbps")

    print("\n✓ Stage 4 完成！")
    print("  关键理解：RVQ 通过多层递进量化残差，用多个小码本实现高精度。")
    print("  可变层数 → 可变比特率（质量/带宽权衡）。")
    print("  下一步 → Stage 5: 卷积编解码器 — SEANet 风格的专业架构")
