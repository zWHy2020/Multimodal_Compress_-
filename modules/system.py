"""多模态 JSCC 主系统编排模块。"""

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .channel_models import BaseChannel, DefaultChannel
from .video_codec import BaseVideoDecoder, BaseVideoEncoder, DefaultVideoDecoder, DefaultVideoEncoder

from .api import ModelForwardOutput
from .depth_codec import DepthJSCCDecoder, DepthJSCCEncoder
from .fusion import JointEntropyModel, JointLatentFusion, MineEstimator


class DepthVideoJSCC(nn.Module):
    """深度图+视频双模态 JSCC（联合压缩版）。"""

    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 4,
        depth_output_dim: int = 128,
        video_hidden_dim: int = 256,
        video_num_frames: int = 5,
        video_output_dim: int = 256,
        shared_latent_dim: int = 128,
        channel_type: str = "awgn",
        snr_db: float = 10.0,
        power_normalization: bool = True,
        enable_omib_stats: bool = True,
        omib_eps: float = 1e-6,
        enable_mi_correction: bool = True,
        mine_hidden_dim: int = 128,
        video_encoder: Optional[BaseVideoEncoder] = None,
        video_decoder: Optional[BaseVideoDecoder] = None,
        channel: Optional[BaseChannel] = None,
    ):
        super().__init__()
        self.depth_encoder = DepthJSCCEncoder(output_dim=depth_output_dim)
        self.depth_decoder = DepthJSCCDecoder(input_dim=depth_output_dim)
        self.video_encoder = video_encoder or DefaultVideoEncoder(
            hidden_dim=video_hidden_dim,
            num_frames=video_num_frames,
            output_dim=video_output_dim,
            use_optical_flow=True,
            use_convlstm=True,
            img_size=img_size,
            patch_size=patch_size,
        )
        self.video_decoder = video_decoder or DefaultVideoDecoder(in_channels=video_output_dim, out_channels=3)

        self.joint_fusion = JointLatentFusion(
            depth_dim=depth_output_dim,
            video_dim=video_output_dim,
            shared_dim=shared_latent_dim,
        )
        self.entropy_model = JointEntropyModel()

        self.channel = channel or DefaultChannel(channel_type=channel_type, snr_db=snr_db, power_normalization=power_normalization)
        self.enable_omib_stats = bool(enable_omib_stats)
        self.omib_eps = float(omib_eps)
        self.enable_mi_correction = bool(enable_mi_correction)
        self.mine_estimator = MineEstimator(depth_output_dim, video_output_dim, hidden_dim=mine_hidden_dim) if self.enable_mi_correction else None

        self.power_normalizer = nn.ModuleDict()
        if power_normalization:
            self.power_normalizer.update({
                'shared': nn.LayerNorm(shared_latent_dim),
                'depth': nn.LayerNorm(depth_output_dim),
                'video': nn.LayerNorm(video_output_dim),
            })

    def _norm_feature(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        if kind not in self.power_normalizer:
            return x
        if kind == 'video':
            v = x.permute(0, 1, 3, 4, 2)
            return self.power_normalizer[kind](v).permute(0, 1, 4, 2, 3)
        if kind == 'depth':
            v = x.permute(0, 2, 3, 1)
            return self.power_normalizer[kind](v).permute(0, 3, 1, 2)
        return self.power_normalizer[kind](x)

    def forward(
        self,
        depth_input: Optional[torch.Tensor] = None,
        video_input: Optional[torch.Tensor] = None,
        snr_db: Optional[float] = None,
    ) -> ModelForwardOutput:
        if snr_db is not None:
            self.channel.set_snr(snr_db)

        out: ModelForwardOutput = {}
        if depth_input is None or video_input is None:
            raise RuntimeError('DepthVideoJSCC 联合压缩需要同时输入 depth_input 和 video_input。')

        depth_encoded, depth_guide = self.depth_encoder(depth_input)
        video_encoded, video_guide = self.video_encoder(video_input)

        fused = self.joint_fusion(depth_encoded, video_encoded)
        entropy_stats = self.entropy_model(
            fused['shared_latent'],
            fused['depth_private'],
            fused['video_private'],
        )

        shared_tx = self.channel(self._norm_feature(fused['shared_latent'], 'shared'))
        depth_private_tx = self.channel(self._norm_feature(fused['depth_private'], 'depth'))
        video_private_tx = self.channel(self._norm_feature(fused['video_private'], 'video'))

        depth_shared_rx = self.joint_fusion.shared_to_depth(shared_tx).unsqueeze(-1).unsqueeze(-1).expand_as(depth_private_tx)
        video_shared_rx = self.joint_fusion.shared_to_video(shared_tx).unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand_as(video_private_tx)

        depth_latent_rx = depth_shared_rx + depth_private_tx
        video_latent_rx = video_shared_rx + video_private_tx

        out['depth_encoded'] = depth_encoded
        out['video_encoded'] = video_encoded
        out['shared_latent'] = fused['shared_latent']
        out['depth_private'] = fused['depth_private']
        out['video_private'] = fused['video_private']
        out['shared_transmitted'] = shared_tx
        out['depth_private_transmitted'] = depth_private_tx
        out['video_private_transmitted'] = video_private_tx

        out['depth_decoded'] = self.depth_decoder(depth_latent_rx, depth_guide)
        out['video_decoded'] = self.video_decoder(
            video_latent_rx,
            video_guide,
            output_size=getattr(self.video_encoder, 'last_input_size', None),
        )

        out['entropy_stats'] = entropy_stats
        out['rate_stats'] = {
            'joint_bpe': entropy_stats['joint_bpe'],
            'shared_bpe': entropy_stats['shared_bpe'],
            'depth_private_bpe': entropy_stats['depth_private_bpe'],
            'video_private_bpe': entropy_stats['video_private_bpe'],
        }

        if self.enable_mi_correction and self.mine_estimator is not None:
            depth_global = fused['depth_private'].mean(dim=(2, 3))
            video_global = fused['video_private'].mean(dim=(1, 3, 4))
            mi_nat = self.mine_estimator(depth_global, video_global)
            mi_bits = (mi_nat / math.log(2.0)).clamp_min(0.0)
            out['rate_stats']['cross_modal_mi_bits'] = mi_bits
            out['entropy_stats']['cross_modal_mi_bits'] = mi_bits

        if self.enable_omib_stats:
            # OMIB-like 变分统计：用私有潜变量的经验高斯参数近似 q_d/q_v，
            # 以便在损失侧实现 KL(q||N(0,I)) 正则。
            depth_mu = fused['depth_private'].mean(dim=(2, 3))
            depth_var = fused['depth_private'].var(dim=(2, 3), unbiased=False).clamp_min(self.omib_eps)
            depth_logvar = torch.log(depth_var)

            video_mu = fused['video_private'].mean(dim=(1, 3, 4))
            video_var = fused['video_private'].var(dim=(1, 3, 4), unbiased=False).clamp_min(self.omib_eps)
            video_logvar = torch.log(video_var)

            depth_kl = 0.5 * (depth_mu.pow(2) + depth_var - 1.0 - depth_logvar)
            video_kl = 0.5 * (video_mu.pow(2) + video_var - 1.0 - video_logvar)

            out['omib_stats'] = {
                'depth_mu': depth_mu,
                'depth_logvar': depth_logvar,
                'video_mu': video_mu,
                'video_logvar': video_logvar,
                'depth_kl': depth_kl.mean(),
                'video_kl': video_kl.mean(),
            }
        return out
