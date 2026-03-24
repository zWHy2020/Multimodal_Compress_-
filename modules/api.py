"""模块间标准数据接口定义。"""

from typing import Any, Dict, Optional, Tuple, TypedDict

import torch


class EncoderOutput(TypedDict):
    latent: torch.Tensor
    guide: torch.Tensor


class FusionOutput(TypedDict):
    shared_latent: torch.Tensor
    depth_shared: torch.Tensor
    video_shared: torch.Tensor
    depth_private: torch.Tensor
    video_private: torch.Tensor


class RateStats(TypedDict, total=False):
    joint_bpe: torch.Tensor
    shared_bpe: torch.Tensor
    depth_private_bpe: torch.Tensor
    video_private_bpe: torch.Tensor
    cross_modal_mi_bits: torch.Tensor


class ModelForwardOutput(TypedDict, total=False):
    mode: str
    depth_encoded: torch.Tensor
    video_encoded: torch.Tensor
    shared_latent: torch.Tensor
    depth_private: torch.Tensor
    video_private: torch.Tensor
    shared_transmitted: torch.Tensor
    depth_private_transmitted: torch.Tensor
    video_private_transmitted: torch.Tensor
    depth_decoded: torch.Tensor
    video_decoded: torch.Tensor
    entropy_stats: Dict[str, torch.Tensor]
    rate_stats: RateStats
    omib_stats: Dict[str, torch.Tensor]


class ModelForwardAPI:
    """统一前向接口约定。"""

    def forward(
        self,
        depth_input: Optional[torch.Tensor],
        video_input: Optional[torch.Tensor],
        snr_db: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> ModelForwardOutput:
        raise NotImplementedError


DepthShape = Tuple[int, int, int, int]
VideoShape = Tuple[int, int, int, int, int]
AnyTensorMap = Dict[str, Any]
