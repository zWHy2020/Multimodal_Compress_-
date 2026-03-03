"""带宽门控相关模块。"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class BandwidthMask(nn.Module):
    """基于带宽比例的通道门控（不改变张量形状）。"""

    def __init__(self, ratio: float = 1.0):
        super().__init__()
        self.ratio = float(ratio)

    def set_ratio(self, ratio: float) -> None:
        self.ratio = float(ratio)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features is None:
            return features
        ratio = float(self.ratio) if self.ratio is not None else 1.0
        ratio = max(0.0, min(1.0, ratio))
        if ratio >= 1.0:
            return features

        if features.dim() == 5:
            channel_dim = 2  # [B, T, C, H, W]
        elif features.dim() == 4:
            channel_dim = 1  # [B, C, H, W]
        elif features.dim() == 3:
            channel_dim = 2  # [B, L, C]
        else:
            return features

        channels = features.size(channel_dim)
        if ratio <= 0.0 or channels == 0:
            return torch.zeros_like(features)

        kept = int(math.ceil(channels * ratio))
        kept = max(1, min(channels, kept))
        mask = torch.zeros(channels, device=features.device, dtype=features.dtype)
        mask[:kept] = 1.0

        if features.dim() == 5:
            mask = mask.view(1, 1, channels, 1, 1)
        elif features.dim() == 4:
            mask = mask.view(1, channels, 1, 1)
        else:
            mask = mask.view(1, 1, channels)
        return features * mask


class ConditionalBandwidthGate(nn.Module):
    """
    条件带宽控制器：使用 (SNR, 带宽比例) 生成 FiLM 风格缩放系数，再执行可选前缀稀疏化。

    说明：
    1) 这里的 gamma(c_ch) 对应报告中的可控调制项，提升“可控性”；
    2) 前缀截断仍保留为工程近似，不宣称语义最优排序。
    """

    def __init__(self, channels: int, hidden_dim: int = 32, ratio: float = 1.0):
        super().__init__()
        self.channels = channels
        self.ratio = float(ratio)
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )

    def set_ratio(self, ratio: float) -> None:
        self.ratio = float(ratio)

    def forward(
        self,
        features: torch.Tensor,
        snr_db: float,
        ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if features is None:
            return features, {}
        ratio_v = self.ratio if ratio is None else float(ratio)
        ratio_v = max(0.0, min(1.0, ratio_v))

        cond = features.new_tensor([[float(snr_db), ratio_v]])
        gamma = torch.sigmoid(self.mlp(cond)).view(1, 1, self.channels, 1, 1)

        if features.dim() != 5:
            # 仅对视频特征 [B,T,C,H,W] 启用条件门控，其他形状直接返回。
            return features, {'bandwidth_gamma_mean': gamma.mean().detach()}

        scaled = features * gamma
        if ratio_v >= 1.0:
            return scaled, {
                'bandwidth_gamma_mean': gamma.mean().detach(),
                'bandwidth_keep_ratio': features.new_tensor(ratio_v),
            }
        if ratio_v <= 0.0:
            return torch.zeros_like(features), {
                'bandwidth_gamma_mean': gamma.mean().detach(),
                'bandwidth_keep_ratio': features.new_tensor(0.0),
            }

        kept = max(1, min(self.channels, int(math.ceil(self.channels * ratio_v))))
        mask = torch.zeros(self.channels, device=features.device, dtype=features.dtype)
        mask[:kept] = 1.0
        masked = scaled * mask.view(1, 1, self.channels, 1, 1)
        return masked, {
            'bandwidth_gamma_mean': gamma.mean().detach(),
            'bandwidth_keep_ratio': features.new_tensor(kept / max(1, self.channels)),
        }
