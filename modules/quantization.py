"""量化模块。"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """轻量 VQ 瓶颈：输出离散索引并统计经验熵（bit/element）。"""

    def __init__(self, dim: int, codebook_size: int = 256, commitment_cost: float = 0.25):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost
        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    @staticmethod
    def _empirical_entropy_bits(indices: torch.Tensor, codebook_size: int) -> torch.Tensor:
        counts = torch.bincount(indices.reshape(-1), minlength=codebook_size).float()
        probs = counts / counts.sum().clamp_min(1.0)
        nz = probs > 0
        entropy = -(probs[nz] * torch.log2(probs[nz])).sum()
        return entropy

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 输入 [B,T,C,H,W]，按最后通道做向量量化。
        flat = x.permute(0, 1, 3, 4, 2).reshape(-1, self.dim)
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=1).unsqueeze(0)
        )
        indices = torch.argmin(distances, dim=1)
        quantized = self.codebook(indices).view(*x.permute(0, 1, 3, 4, 2).shape)
        quantized = quantized.permute(0, 1, 4, 2, 3).contiguous()

        codebook_loss = F.mse_loss(quantized.detach(), x)
        commitment_loss = F.mse_loss(quantized, x.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        quantized = x + (quantized - x).detach()
        entropy_bits = self._empirical_entropy_bits(indices, self.codebook_size).to(x.device)
        return quantized, indices, vq_loss, entropy_bits
