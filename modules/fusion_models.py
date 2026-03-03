"""跨模态融合/熵模型/MI估计抽象与默认实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn

from .fusion import JointEntropyModel, JointLatentFusion, MineEstimator


class BaseJointFusion(nn.Module, ABC):
    """跨模态联合融合抽象接口。"""

    @abstractmethod
    def forward(self, depth_feat: torch.Tensor, video_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """返回共享+私有潜变量字典。"""

    @abstractmethod
    def project_shared_to_depth(self, shared_latent: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """将共享潜变量投影到 depth 私有空间形状。"""

    @abstractmethod
    def project_shared_to_video(self, shared_latent: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """将共享潜变量投影到 video 私有空间形状。"""


class BaseEntropyModel(nn.Module, ABC):
    """熵模型抽象接口。"""

    @abstractmethod
    def forward(self, shared: torch.Tensor, depth_private: torch.Tensor, video_private: torch.Tensor) -> Dict[str, torch.Tensor]:
        """估计 joint/shared/private 码率统计。"""


class BaseMineEstimator(nn.Module, ABC):
    """互信息估计器抽象接口。"""

    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """返回互信息下界（nat）。"""


class DefaultJointFusion(BaseJointFusion):
    """默认融合实现：封装 JointLatentFusion。"""

    def __init__(self, depth_dim: int, video_dim: int, shared_dim: int = 128):
        super().__init__()
        self.impl = JointLatentFusion(depth_dim=depth_dim, video_dim=video_dim, shared_dim=shared_dim)

    def forward(self, depth_feat: torch.Tensor, video_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.impl(depth_feat, video_feat)

    def project_shared_to_depth(self, shared_latent: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return self.impl.shared_to_depth(shared_latent).unsqueeze(-1).unsqueeze(-1).expand_as(like)

    def project_shared_to_video(self, shared_latent: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return self.impl.shared_to_video(shared_latent).unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand_as(like)


class DefaultEntropyModel(BaseEntropyModel):
    """默认熵模型实现：封装 JointEntropyModel。"""

    def __init__(self):
        super().__init__()
        self.impl = JointEntropyModel()

    def forward(self, shared: torch.Tensor, depth_private: torch.Tensor, video_private: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.impl(shared, depth_private, video_private)


class DefaultMineEstimator(BaseMineEstimator):
    """默认 MI 估计器：封装 MineEstimator。"""

    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.impl = MineEstimator(x_dim=x_dim, y_dim=y_dim, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.impl(x, y)
