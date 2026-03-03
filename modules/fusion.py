"""跨模态融合、熵模型与互信息估计模块。"""

import math
from typing import Dict

import torch
import torch.nn as nn


class JointLatentFusion(nn.Module):
    """共享潜变量 + 私有残差分解模块。"""

    def __init__(self, depth_dim: int, video_dim: int, shared_dim: int = 128):
        super().__init__()
        self.shared_dim = shared_dim
        self.fuser = nn.Sequential(
            nn.Linear(depth_dim + video_dim, shared_dim),
            nn.GELU(),
            nn.Linear(shared_dim, shared_dim),
        )
        self.shared_to_depth = nn.Linear(shared_dim, depth_dim)
        self.shared_to_video = nn.Linear(shared_dim, video_dim)

    def forward(self, depth_feat: torch.Tensor, video_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        depth_global = depth_feat.mean(dim=(2, 3))
        video_global = video_feat.mean(dim=(1, 3, 4))
        shared = self.fuser(torch.cat([depth_global, video_global], dim=-1))

        depth_from_shared = self.shared_to_depth(shared).unsqueeze(-1).unsqueeze(-1).expand_as(depth_feat)
        video_from_shared = self.shared_to_video(shared).unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand_as(video_feat)

        depth_private = depth_feat - depth_from_shared
        video_private = video_feat - video_from_shared

        return {
            'shared_latent': shared,
            'depth_shared': depth_from_shared,
            'video_shared': video_from_shared,
            'depth_private': depth_private,
            'video_private': video_private,
        }


class JointEntropyModel(nn.Module):
    """联合熵模型（高斯先验近似）用于估计共享/私有码率。"""

    def __init__(self):
        super().__init__()
        self.log_scale = nn.ParameterDict({
            'shared': nn.Parameter(torch.tensor(0.0)),
            'depth_private': nn.Parameter(torch.tensor(0.0)),
            'video_private': nn.Parameter(torch.tensor(0.0)),
        })

    def _nll_bits(self, x: torch.Tensor, key: str) -> torch.Tensor:
        scale = torch.exp(self.log_scale[key]).clamp_min(1e-4)
        nll_nat = 0.5 * ((x / scale) ** 2) + torch.log(scale) + 0.5 * math.log(2 * math.pi)
        return nll_nat / math.log(2.0)

    def forward(self, shared: torch.Tensor, depth_private: torch.Tensor, video_private: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_bits = self._nll_bits(shared, 'shared').mean()
        depth_bits = self._nll_bits(depth_private, 'depth_private').mean()
        video_bits = self._nll_bits(video_private, 'video_private').mean()
        total = shared_bits + depth_bits + video_bits
        return {
            'shared_bpe': shared_bits,
            'depth_private_bpe': depth_bits,
            'video_private_bpe': video_bits,
            'joint_bpe': total,
        }


class MineEstimator(nn.Module):
    """MINE: 估计 I(X;Y) 的神经下界（Donsker-Varadhan 形式）。"""

    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or y.dim() != 2:
            raise RuntimeError('MineEstimator expects 2D tensors [B, D].')
        if x.size(0) != y.size(0):
            raise RuntimeError('MineEstimator expects matched batch size for x and y.')
        joint = torch.cat([x, y], dim=-1)
        y_perm = y[torch.randperm(y.size(0), device=y.device)]
        marginal = torch.cat([x, y_perm], dim=-1)
        t_joint = self.net(joint)
        t_marginal = self.net(marginal)
        mi_nat = t_joint.mean() - torch.log(torch.exp(t_marginal).mean().clamp_min(1e-8))
        return mi_nat
