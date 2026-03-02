"""
多模态联合信源信道编码（JSCC）主模型

整合深度图与视频双通路的编码器、信道和解码器。
实现双模态端到端 JSCC 训练。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math

from video_encoder import VideoJSCCEncoder, VideoJSCCDecoder
from video_unet import VideoUNetDecoder
from channel import Channel


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

    def forward(self, features: torch.Tensor, snr_db: float, ratio: Optional[float] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
    ):
        super().__init__()
        self.depth_encoder = DepthJSCCEncoder(output_dim=depth_output_dim)
        self.depth_decoder = DepthJSCCDecoder(input_dim=depth_output_dim)
        self.video_encoder = VideoJSCCEncoder(
            hidden_dim=video_hidden_dim,
            num_frames=video_num_frames,
            output_dim=video_output_dim,
            use_optical_flow=True,
            use_convlstm=True,
            img_size=img_size,
            patch_size=patch_size,
        )
        self.video_decoder = VideoUNetDecoder(in_channels=video_output_dim, out_channels=3)

        self.joint_fusion = JointLatentFusion(
            depth_dim=depth_output_dim,
            video_dim=video_output_dim,
            shared_dim=shared_latent_dim,
        )
        self.entropy_model = JointEntropyModel()

        self.channel = Channel(channel_type=channel_type, snr_db=snr_db, power_normalization=power_normalization)
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
    ) -> Dict[str, Any]:
        if snr_db is not None:
            self.channel.set_snr(snr_db)

        out: Dict[str, Any] = {}
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
            semantic_context=None,
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
