"""视频编解码模块抽象与默认实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn

from video_encoder import VideoJSCCEncoder
from video_unet import VideoUNetDecoder


class BaseVideoEncoder(nn.Module, ABC):
    """视频编码器抽象接口。"""

    @abstractmethod
    def forward(self, video_input: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """返回 (编码特征, 引导向量)。"""


class BaseVideoDecoder(nn.Module, ABC):
    """视频解码器抽象接口。"""

    @abstractmethod
    def forward(
        self,
        noisy_features: torch.Tensor,
        guide_vectors: Optional[torch.Tensor] = None,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """根据潜变量重建视频。"""


class DefaultVideoEncoder(BaseVideoEncoder):
    """默认视频编码器：对现有 VideoJSCCEncoder 的薄封装。"""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_frames: int = 5,
        output_dim: int = 256,
        use_optical_flow: bool = True,
        use_convlstm: bool = True,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 4,
    ):
        super().__init__()
        self.impl = VideoJSCCEncoder(
            hidden_dim=hidden_dim,
            num_frames=num_frames,
            output_dim=output_dim,
            use_optical_flow=use_optical_flow,
            use_convlstm=use_convlstm,
            img_size=img_size,
            patch_size=patch_size,
        )

    @property
    def last_input_size(self):
        return getattr(self.impl, 'last_input_size', None)

    def forward(self, video_input: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.impl(video_input)


class DefaultVideoDecoder(BaseVideoDecoder):
    """默认视频解码器：对现有 VideoUNetDecoder 的薄封装。"""

    def __init__(self, in_channels: int = 256, out_channels: int = 3):
        super().__init__()
        self.impl = VideoUNetDecoder(in_channels=in_channels, out_channels=out_channels)

    def forward(
        self,
        noisy_features: torch.Tensor,
        guide_vectors: Optional[torch.Tensor] = None,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        return self.impl(noisy_features, guide_vectors, output_size=output_size)

