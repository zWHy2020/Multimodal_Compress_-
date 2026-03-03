"""深度模态编码/解码模块。"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class DepthJSCCEncoder(nn.Module):
    """轻量深度图编码器，将单通道深度图映射到信道潜变量。"""

    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, output_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, depth_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.net(depth_input)
        guide = encoded.mean(dim=(2, 3))
        return encoded, guide


class DepthJSCCDecoder(nn.Module):
    """轻量深度图解码器。"""

    def __init__(self, input_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, encoded: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.net(encoded)
