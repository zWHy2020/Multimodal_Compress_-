"""信道模块抽象与默认实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from channel import Channel


class BaseChannel(nn.Module, ABC):
    """可插拔信道抽象接口。"""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行信道传输。"""

    @abstractmethod
    def set_snr(self, snr_db: float):
        """动态更新 SNR。"""


class DefaultChannel(BaseChannel):
    """默认信道实现：兼容现有 Channel。"""

    def __init__(
        self,
        channel_type: str = "awgn",
        snr_db: float = 10.0,
        power_normalization: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.impl = Channel(
            channel_type=channel_type,
            snr_db=snr_db,
            power_normalization=power_normalization,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.impl(x)

    def set_snr(self, snr_db: float):
        self.impl.set_snr(snr_db)

