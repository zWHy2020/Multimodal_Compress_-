"""深度编解码抽象与默认实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .depth_codec import DepthJSCCDecoder, DepthJSCCEncoder
from .depth_deepjscc import DeepJSCCDepthDecoder, DeepJSCCDepthEncoder


class BaseDepthEncoder(nn.Module, ABC):
    """深度编码器抽象接口。"""

    @abstractmethod
    def forward(self, depth_input: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """返回 (编码特征, 引导向量)。"""


class BaseDepthDecoder(nn.Module, ABC):
    """深度解码器抽象接口。"""

    @abstractmethod
    def forward(self, encoded: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        """根据潜变量重建深度图。"""


class DefaultDepthEncoder(BaseDepthEncoder):
    """默认深度编码器：封装现有 DepthJSCCEncoder。"""

    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.impl = DepthJSCCEncoder(output_dim=output_dim)

    def forward(self, depth_input: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.impl(depth_input)


class DefaultDepthDecoder(BaseDepthDecoder):
    """默认深度解码器：封装现有 DepthJSCCDecoder。"""

    def __init__(self, input_dim: int = 128):
        super().__init__()
        self.impl = DepthJSCCDecoder(input_dim=input_dim)

    def forward(self, encoded: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.impl(encoded, guide)


class ExternalDeepJSCCDepthEncoder(BaseDepthEncoder):
    """External DeepJSCC-style encoder adapted for single-channel depth."""

    def __init__(self, c: int = 64, power_constraint: float = 1.0):
        super().__init__()
        self.impl = DeepJSCCDepthEncoder(c=c, power_constraint=power_constraint)

    def forward(self, depth_input: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.impl(depth_input)


class ExternalDeepJSCCDepthDecoder(BaseDepthDecoder):
    """External DeepJSCC-style decoder adapted for single-channel depth."""

    def __init__(self, c: int = 64):
        super().__init__()
        self.impl = DeepJSCCDepthDecoder(c=c)

    def forward(self, encoded: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.impl(encoded, guide)
