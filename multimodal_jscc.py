"""
多模态联合信源信道编码（JSCC）主模型

整合文本、图像、视频三个通路的编码器、信道和解码器。
实现跨模态交叉注意力引导机制和端到端训练。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math

from text_encoder import TextJSCCEncoder, TextJSCCDecoder
from image_encoder import ImageJSCCEncoder, ImageJSCCDecoder
from generative_image_decoder import GenerativeImageJSCCDecoder
from generator_adapter import VAEGeneratorAdapter
from video_encoder import VideoJSCCEncoder, VideoJSCCDecoder
from video_unet import VideoUNetDecoder
from cross_attention import MultiModalCrossAttention
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


class MultimodalJSCC(nn.Module):
    """
    多模态联合信源信道编码模型
    
    整合三个模态的编码器、信道和解码器，实现跨模态交叉注意力引导。
    支持文本、图像、视频的并行JSCC传输。
    """
    
    def __init__(
        self,
        # 文本编码器参数
        vocab_size: int = 10000,
        text_embed_dim: int = 512,
        text_num_heads: int = 8,
        text_num_layers: int = 6,
        text_output_dim: int = 256,
        
        # 图像编码器参数
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 4,
        img_embed_dims: List[int] = [96, 192, 384, 768],
        img_depths: List[int] = [2, 2, 6, 2],
        img_num_heads: List[int] = [3, 6, 12, 24],
        img_output_dim: int = 256,
        img_window_size: int = 7,
        mlp_ratio: float = 4.0,
        pretrained: bool = False,  # 【Phase 1】是否使用预训练权重
        freeze_encoder: bool = False,  # 【Phase 1】是否冻结编码器主干
        pretrained_model_name: str = 'swin_tiny_patch4_window7_224',
        image_decoder_type: str = "baseline",
        generator_type: str = "vae",
        generator_ckpt: Optional[str] = None,
        z_channels: int = 4,
        latent_down: int = 8,
        
        
        # 视频编码器参数
        video_hidden_dim: int = 256,
        video_num_frames: int = 5,
        video_clip_len: int | None = None,
        video_use_optical_flow: bool = True,
        video_use_convlstm: bool = True,
        video_output_dim: int = 256,
        video_gop_size: Optional[int] = None,
        video_latent_downsample_factor: int = 2,
        video_latent_downsample_stride: Optional[int] = None,
        video_decoder_type: str = "unet",
        video_unet_base_channels: int = 64,
        video_unet_num_down: int = 4,
        video_unet_num_res_blocks: int = 3,
        video_decode_chunk_size: Optional[int] = None,
        video_entropy_max_exact_quantile_elems: int = 2_000_000,
        video_entropy_quantile_sample_size: int = 262_144,
        
        # 信道参数
        channel_type: str = "awgn",
        snr_db: float = 10.0,
        power_normalization: bool = True,
        use_quantization_noise: bool = False,
        quantization_noise_range: float = 0.5,
        normalize_inputs: bool = False,
        use_text_guidance_image: bool = False,
        use_text_guidance_video: bool = False,
        enforce_text_condition: bool = True,
        condition_margin_weight: float = 0.1,
        condition_margin: float = 0.05,
        condition_prob: float = 0.5,
        condition_only_low_snr: bool = True,
        condition_low_snr_threshold: float = 5.0,
        use_gradient_checkpointing: bool = True,
        use_conditional_bandwidth_control: bool = True,
        bandwidth_condition_hidden: int = 32,
        enable_vq_bottleneck: bool = False,
        vq_codebook_size: int = 256,
        vq_commitment_cost: float = 0.25,
        
        # 训练参数
    ):
        super().__init__()

        if video_clip_len is not None:
            video_num_frames = video_clip_len
        
        # 【Phase 1】保存预训练参数
        self.pretrained = pretrained
        self.freeze_encoder = freeze_encoder
        self.image_decoder_type = image_decoder_type
        self.generator_type = generator_type
        self.generator_ckpt = generator_ckpt
        self.z_channels = z_channels
        self.latent_down = latent_down
        
        # 文本编码器和解码器
        self.text_encoder = TextJSCCEncoder(
            vocab_size=vocab_size,
            d_model=text_embed_dim,
            num_heads=text_num_heads,
            num_layers=text_num_layers,
            output_dim=text_output_dim
        )
        self.text_decoder = TextJSCCDecoder(
            vocab_size=vocab_size,
            d_model=text_embed_dim,
            num_heads=text_num_heads,
            num_layers=text_num_layers,
            input_dim=text_output_dim
        )
        
        # 【Phase 1】图像编码器和解码器（支持预训练权重）
        pretrained = getattr(self, 'pretrained', False)
        freeze_encoder = getattr(self, 'freeze_encoder', False)
        
        self.image_encoder = ImageJSCCEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dims=img_embed_dims,
            depths=img_depths,
            num_heads=img_num_heads,
            window_size=img_window_size,
            mlp_ratio=mlp_ratio,
            output_dim=img_output_dim,
            pretrained=pretrained,  # 【Phase 1】是否使用预训练权重
            freeze_encoder=freeze_encoder,  # 【Phase 1】是否冻结编码器主干
            pretrained_model_name=pretrained_model_name,
            use_gradient_checkpointing=use_gradient_checkpointing,
            # 不传入 patch_embed, swin_layers, swin_norm，让编码器使用独立路径（768维）
        )
        if image_decoder_type.lower() == "generative":
            generator_adapter = VAEGeneratorAdapter(
                z_channels=z_channels,
                latent_down=latent_down,
                generator_type=generator_type,
                pretrained_path=generator_ckpt,
            )
            self.image_decoder = GenerativeImageJSCCDecoder(
                img_size=img_size,
                patch_size=patch_size,
                embed_dims=img_embed_dims,
                depths=img_depths,
                num_heads=img_num_heads,
                window_size=img_window_size,
                mlp_ratio=mlp_ratio,
                input_dim=img_output_dim,
                semantic_context_dim=text_output_dim,
                normalize_output=normalize_inputs,
                use_gradient_checkpointing=use_gradient_checkpointing,
                z_channels=z_channels,
                latent_down=latent_down,
                generator_adapter=generator_adapter,
            )
        else:
            self.image_decoder = ImageJSCCDecoder(
                img_size=img_size,
                patch_size=patch_size,
                embed_dims=img_embed_dims,
                depths=img_depths,
                num_heads=img_num_heads,
                window_size=img_window_size,
                mlp_ratio=mlp_ratio,
                input_dim=img_output_dim,
                semantic_context_dim=text_output_dim,  # 【修复】传入语义上下文维度，与 VideoJSCCDecoder 保持一致
                normalize_output=normalize_inputs,
                use_gradient_checkpointing=use_gradient_checkpointing,
                # 编码器和解码器都使用 embed_dims，维度完全匹配，无需特殊 guide_dim
            )
        
        if video_latent_downsample_stride is not None:
            video_latent_downsample_factor = video_latent_downsample_stride

        # 视频编码器和解码器（独立主干，使用 video_hidden_dim=256）
        self.video_encoder = VideoJSCCEncoder(
            hidden_dim=video_hidden_dim,
            num_frames=video_num_frames,
            use_optical_flow=video_use_optical_flow,
            use_convlstm=video_use_convlstm,
            output_dim=video_output_dim,
            gop_size=video_gop_size,
            latent_downsample_factor=video_latent_downsample_factor,
            mlp_ratio=mlp_ratio,
            img_size=img_size,
            patch_size=patch_size,
            use_gradient_checkpointing=use_gradient_checkpointing,
            entropy_max_exact_quantile_elems=video_entropy_max_exact_quantile_elems,
            entropy_quantile_sample_size=video_entropy_quantile_sample_size,
            # 不传入 patch_embed, swin_layers, swin_norm，让编码器使用独立路径
        )
        if video_decoder_type.lower() == "swin":
            self.video_decoder = VideoJSCCDecoder(
                hidden_dim=video_hidden_dim,
                num_frames=video_num_frames,
                use_optical_flow=video_use_optical_flow,
                use_convlstm=video_use_convlstm,
                input_dim=video_output_dim,
                img_size=img_size,  # 添加图像尺寸参数，用于上采样
                patch_size=patch_size,  # 添加 patch 大小参数，用于上采样
                semantic_context_dim=text_output_dim,  # 添加语义上下文维度，用于语义对齐层
                normalize_output=normalize_inputs,
                latent_upsample_factor=video_latent_downsample_factor,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
        else:
            self.video_decoder = VideoUNetDecoder(
                in_channels=video_output_dim,
                out_channels=3,
                base_channels=video_unet_base_channels,
                num_down=video_unet_num_down,
                num_res_blocks=video_unet_num_res_blocks,
                use_tanh=True,
                normalize_output=normalize_inputs,
                decode_chunk_size=video_decode_chunk_size,
            )
        
        # 信道模型
        self.channel = Channel(
            channel_type=channel_type,
            snr_db=snr_db,
            power_normalization=power_normalization
        )
        self.use_quantization_noise = use_quantization_noise
        self.quantization_noise_range = quantization_noise_range
        self.normalize_inputs = normalize_inputs
        self.use_text_guidance_image = use_text_guidance_image
        self.use_text_guidance_video = use_text_guidance_video
        self.enforce_text_condition = enforce_text_condition
        self.condition_margin_weight = condition_margin_weight
        self.condition_margin = condition_margin
        self.condition_prob = condition_prob
        self.condition_only_low_snr = condition_only_low_snr
        self.condition_low_snr_threshold = condition_low_snr_threshold
        self.bandwidth_ratio = 1.0
        self.bandwidth_mask = BandwidthMask(self.bandwidth_ratio)
        self.use_conditional_bandwidth_control = use_conditional_bandwidth_control
        self.conditional_bandwidth_gate = ConditionalBandwidthGate(
            channels=video_output_dim,
            hidden_dim=bandwidth_condition_hidden,
            ratio=self.bandwidth_ratio,
        )
        self.enable_vq_bottleneck = enable_vq_bottleneck
        self.vq_bottleneck = VectorQuantizer(
            dim=video_output_dim,
            codebook_size=vq_codebook_size,
            commitment_cost=vq_commitment_cost,
        ) if enable_vq_bottleneck else None
        
        # 功率归一化模块（可选）
        # 始终创建属性以便在评估/推理阶段安全访问
        self.power_normalizer = nn.ModuleDict()
        if power_normalization:
            self.power_normalizer.update({
                'text': nn.LayerNorm(text_output_dim),
                'image': nn.LayerNorm(img_output_dim),
                'video': nn.LayerNorm(video_output_dim)
            })
    
    def forward(
        self,
        text_input: Optional[torch.Tensor] = None,
        image_input: Optional[torch.Tensor] = None,
        video_input: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        snr_db: float = 10.0
    ) -> Dict[str, Any]:
        """
        多模态JSCC前向传播 - 鲁棒性增强版本
        
        Args:
            text_input (torch.Tensor, optional): 文本输入 [B, seq_len]
            image_input (torch.Tensor, optional): 图像输入 [B, C, H, W]
            video_input (torch.Tensor, optional): 视频输入 [B, T, C, H, W]
            text_attention_mask (torch.Tensor, optional): 文本注意力掩码
            snr_db (float, optional): 信噪比
            
        Returns:
            Dict[str, Any]: 包含编码特征、解码结果等的字典
        """
        results = {}
        
        # 设置信噪比
        if snr_db is not None:
            self.channel.set_snr(snr_db)
        
        # 鲁棒性检查：确保输入不为None且形状正确
        text_input = self._validate_input(text_input, 'text')
        image_input = self._validate_input(image_input, 'image')
        video_input = self._validate_input(video_input, 'video')
        if self.use_text_guidance_video and video_input is not None and text_input is None:
            raise RuntimeError("use_text_guidance_video=True 但缺少文本输入，无法提供语义条件。")
        
        # 编码阶段
        encoded_features = {}
        guide_vectors = {}
        text_encoded = None
        text_encoded_for_semantic = None  # 用于语义上下文的原始文本编码（未归一化）
        
        # 文本编码
        if text_input is not None:
            text_encoded, text_guide = self.text_encoder(text_input, text_attention_mask)
            # 克隆一份原始文本编码，用于后续的语义上下文（避免被归一化修改）
            text_encoded_for_semantic = text_encoded.clone()
            encoded_features['text'] = text_encoded
            guide_vectors['text'] = text_guide
            results['text_encoded'] = text_encoded
            results['text_guide'] = text_guide
        
        # 图像编码
        if image_input is not None:
            # 输入形状验证
            assert image_input.dim() == 4, (
                f"图像输入应为4D张量 [B, C, H, W]，实际为 {image_input.dim()}D，shape={tuple(image_input.shape)}"
            )
            expected_img_hw = None
            
            # 调用编码器
            image_encoded, image_guide = self.image_encoder(image_input, snr_db=snr_db)
            
            # 编码后形状验证
            patch_size = None
            if hasattr(self.image_encoder, 'patch_embed') and hasattr(self.image_encoder.patch_embed, 'patch_size'):
                _ps = self.image_encoder.patch_embed.patch_size
                if isinstance(_ps, (tuple, list)):
                    patch_size = int(_ps[0])
                else:
                    patch_size = int(_ps)
            #if (expected_img_hw is not None) and (patch_size is not None):
                #expected_L = (expected_img_hw[0] // patch_size) * (expected_img_hw[1] // patch_size)
                #assert image_encoded.shape[1] == expected_L, (
                    #f"图像编码后序列长度不匹配: got L={image_encoded.shape[1]}, expected={expected_L} (img={expected_img_hw}, patch={patch_size})"
                #)
            
            # 验证输出维度
            #expected_img_dim = getattr(self.power_normalizer['image'], 'normalized_shape', (None,))
            #expected_img_dim = expected_img_dim[0] if isinstance(expected_img_dim, (list, tuple)) else expected_img_dim
            #if expected_img_dim is not None:
                #assert image_encoded.shape[-1] == expected_img_dim, (
                    #f"图像编码输出维度不匹配: got {image_encoded.shape[-1]}, expected={expected_img_dim}"
                #)
            
            encoded_features['image'] = image_encoded
            guide_vectors['image'] = image_guide
            results['image_encoded'] = image_encoded
            results['image_guide'] = image_guide
        
        # 视频编码
        if video_input is not None:
            video_encoded, video_guide = self.video_encoder(video_input, snr_db=snr_db)
            # 说明：条件门控先执行 gamma(c_ch) 缩放，再执行可选前缀稀疏化。
            # 这样保留了带宽比率控制接口，同时引入对信道状态(SNR)的可控条件化能力。
            gate_stats = {}
            if self.use_conditional_bandwidth_control:
                video_encoded, gate_stats = self.conditional_bandwidth_gate(video_encoded, snr_db=snr_db, ratio=self.bandwidth_ratio)
            else:
                video_encoded = self.bandwidth_mask(video_encoded)

            if self.vq_bottleneck is not None:
                # 说明：VQ 输出离散索引，可统计经验熵(bit/element)用于量化分析。
                video_encoded, vq_indices, vq_loss, vq_entropy_bits = self.vq_bottleneck(video_encoded)
                results['video_vq_indices'] = vq_indices
                results['video_vq_loss'] = vq_loss
                results['video_empirical_entropy_bits'] = vq_entropy_bits

            results.update(gate_stats)
            # 验证输出维度
            #expected_video_dim = getattr(self.power_normalizer['video'], 'normalized_shape', (None,))
            #expected_video_dim = expected_video_dim[0] if isinstance(expected_video_dim, (list, tuple)) else expected_video_dim
            #if expected_video_dim is not None and video_encoded.dim() == 5:
                #assert video_encoded.size(2) == expected_video_dim, (
                    #f"视频编码输出维度不匹配: got {video_encoded.size(2)}, expected={expected_video_dim}"
                #)
            encoded_features['video'] = video_encoded
            guide_vectors['video'] = video_guide
            results['video_encoded'] = video_encoded
            results['video_guide'] = video_guide
        results['bandwidth_ratio'] = self.bandwidth_ratio
        
        # 功率归一化
        for modality, features in encoded_features.items():
            if modality in self.power_normalizer:
                if modality == 'video':
                    # 视频特征形状为 [B, T, C, H, W]，需要将归一化维度放到最后
                    v = features.permute(0, 1, 3, 4, 2)
                    v = self.power_normalizer[modality](v)
                    encoded_features[modality] = v.permute(0, 1, 4, 2, 3)
                else:
                    encoded_features[modality] = self.power_normalizer[modality](features)
        
        # 信道传输
        transmitted_features = {}
        for modality, features in encoded_features.items():
            features = self._apply_quantization_noise(features)
            transmitted_features[modality] = self.channel(features)
            results[f'{modality}_transmitted'] = transmitted_features[modality]
        # 说明：energy_rate_proxy 是能量代理，不等价于离散比特流；
        # 若启用 VQ，则额外记录经验熵 empirical_entropy_bits 以提升“可量化性”。
        rate_stats = {
            f'{modality}_energy_rate_proxy': features.pow(2).mean()
            for modality, features in encoded_features.items()
        }
        if 'video_empirical_entropy_bits' in results:
            rate_stats['video_empirical_entropy_bits'] = results['video_empirical_entropy_bits']
            rate_stats['video_vq_upper_bound_bits'] = torch.log2(
                torch.tensor(float(self.vq_bottleneck.codebook_size), device=results['video_empirical_entropy_bits'].device)
            )
        if hasattr(self.video_encoder, "last_rate_stats") and self.video_encoder.last_rate_stats:
            rate_stats.update(self.video_encoder.last_rate_stats)
        results['rate_stats'] = rate_stats
        
        # 解码阶段 - 语义引导式解码
        decoded_outputs = {}
        
        # 文本解码（不需要语义引导）
        if 'text' in transmitted_features and text_input is not None:
            text_decoded = self.text_decoder(
                transmitted_features['text'],
                guide_vectors['text'],
                text_attention_mask
            )
            decoded_outputs['text'] = text_decoded
            results['text_decoded'] = text_decoded
        
        # 图像解码（使用文本语义引导）
        if 'image' in transmitted_features and image_input is not None:
            semantic_for_image = text_encoded_for_semantic if self.use_text_guidance_image else None
            image_decoded = self.image_decoder(
                transmitted_features['image'],
                guide_vectors['image'],
                semantic_context=semantic_for_image,  # 路线1默认禁用 text->image
                snr_db=snr_db,
                input_resolution=getattr(self.image_encoder, "last_latent_resolution", None),
                output_resolution=getattr(self.image_encoder, "last_patch_resolution", None),
                output_size=getattr(self.image_encoder, "last_input_size", None),
            )
            decoded_outputs['image'] = image_decoded
            results['image_decoded'] = image_decoded
        
        # 视频解码（使用文本语义引导）
        if 'video' in transmitted_features and video_input is not None:
            semantic_for_video = text_encoded_for_semantic if self.use_text_guidance_video else None
            if self.use_text_guidance_video and semantic_for_video is None:
                raise RuntimeError("视频解码需要语义上下文，但文本编码缺失。")
            video_decoded = self.video_decoder(
                transmitted_features['video'],
                guide_vectors['video'],
                semantic_context=semantic_for_video,  # None 时视频解码不会走语义 cross-attention
                output_size=getattr(self.video_encoder, "last_input_size", None),
            )
            decoded_outputs['video'] = video_decoded
            results['video_decoded'] = video_decoded
            if getattr(self.video_decoder, "last_semantic_gate_stats", None):
                stats = self.video_decoder.last_semantic_gate_stats
                results["video_semantic_gate_mean"] = stats.get("mean")
                results["video_semantic_gate_std"] = stats.get("std")
        
        results.update(decoded_outputs)
        return results

    def _apply_quantization_noise(self, features: torch.Tensor) -> torch.Tensor:
        if not self.use_quantization_noise or not self.training:
            return features
        if self.quantization_noise_range <= 0:
            return features
        noise = torch.empty_like(features).uniform_(
            -self.quantization_noise_range,
            self.quantization_noise_range
        )
        return features + noise
    
    def _validate_input(self, input_tensor: Optional[torch.Tensor], modality: str) -> Optional[torch.Tensor]:
        """
        验证输入张量的有效性
        
        Args:
            input_tensor: 输入张量
            modality: 模态类型
            
        Returns:
            Optional[torch.Tensor]: 验证后的张量或None
        """
        if input_tensor is None:
            return None
        
        # 检查张量是否包含有效数据
        if input_tensor.numel() == 0:
            print(f"警告: {modality}输入为空张量")
            return input_tensor
        
        # 检查是否全为零（可能是填充的零张量），仅提示
        if torch.all(input_tensor == 0):
            print(f"警告: {modality}输入全为零，可能是填充数据或缺失模态占位")
            return input_tensor
        
        # 检查形状
        if modality == 'text' and input_tensor.dim() != 2:
            print(f"警告: {modality}输入形状不正确，期望2D，得到{input_tensor.dim()}D")
            return None
        elif modality == 'image' and input_tensor.dim() != 4:
            print(f"警告: {modality}输入形状不正确，期望4D，得到{input_tensor.dim()}D")
            return None
        elif modality == 'video':
            if input_tensor.dim() != 5:
                print(f"警告: {modality}输入形状不正确，期望5D，得到{input_tensor.dim()}D")
                return None
            # 额外鲁棒性：若误传为 [B, C, T, H, W]，自动修正为 [B, T, C, H, W]
            # 依据常见通道数判断（C∈{1,3}）且时间维通常 > 1
            B, D1, D2, D3, D4 = input_tensor.shape
            if D1 in (1, 3) and D2 not in (1, 3):
                print(f"警告: 检测到视频张量可能为 [B, C, T, H, W]，将自动重排为 [B, T, C, H, W]。原形状: {tuple(input_tensor.shape)}")
                input_tensor = input_tensor.permute(0, 2, 1, 3, 4).contiguous()
        
        return input_tensor
    
    
    def encode(
        self,
        text_input: Optional[torch.Tensor] = None,
        image_input: Optional[torch.Tensor] = None,
        video_input: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        snr_db: float = 10.0
    ) -> Dict[str, torch.Tensor]:
        """
        仅编码阶段
        
        Args:
            text_input, image_input, video_input: 各模态输入
            text_attention_mask: 文本注意力掩码
            
        Returns:
            Dict[str, torch.Tensor]: 编码特征和引导向量
        """
        results = {}
        
        # 文本编码
        if text_input is not None:
            text_encoded, text_guide = self.text_encoder(text_input, text_attention_mask)
            results['text_encoded'] = text_encoded
            results['text_guide'] = text_guide
        
        # 图像编码
        if image_input is not None:
            image_encoded, image_guide = self.image_encoder(image_input, snr_db=snr_db)
            results['image_encoded'] = image_encoded
            results['image_guide'] = image_guide
        
        # 视频编码
        if video_input is not None:
            video_encoded, video_guide = self.video_encoder(video_input, snr_db=snr_db)
            if self.use_conditional_bandwidth_control:
                video_encoded, _ = self.conditional_bandwidth_gate(video_encoded, snr_db=snr_db, ratio=self.bandwidth_ratio)
            else:
                video_encoded = self.bandwidth_mask(video_encoded)
            if self.vq_bottleneck is not None:
                video_encoded, _, _, _ = self.vq_bottleneck(video_encoded)
            results['video_encoded'] = video_encoded
            results['video_guide'] = video_guide
        
        return results
    
    def decode(
        self,
        transmitted_features: Dict[str, torch.Tensor],
        guide_vectors: Dict[str, torch.Tensor],
        text_attention_mask: Optional[torch.Tensor] = None,
        semantic_context: Optional[torch.Tensor] = None,
        multiple_semantic_contexts: Optional[List[torch.Tensor]] = None,
        image_input_resolution: Optional[Tuple[int, int]] = None,
        image_output_resolution: Optional[Tuple[int, int]] = None,
        snr_db: float = 10.0
    ) -> Dict[str, torch.Tensor]:
        """
        仅解码阶段
        
        Args:
            transmitted_features: 传输后的特征
            guide_vectors: 引导向量
            text_attention_mask: 文本注意力掩码
            semantic_context: 语义上下文（文本编码），用于图像和视频解码的语义引导
                            如果为None，则图像和视频解码将不使用语义引导
            
        Returns:
            Dict[str, torch.Tensor]: 解码输出
        """
        results = {}
        
        # 文本解码
        if 'text' in transmitted_features:
            text_decoded = self.text_decoder(
                transmitted_features['text'],
                guide_vectors['text'],
                text_attention_mask
            )
            results['text_decoded'] = text_decoded
        
        # 图像解码（使用文本语义引导，如果提供）
        if 'image' in transmitted_features:
            semantic_for_image = semantic_context if self.use_text_guidance_image else None
            image_decoded = self.image_decoder(
                transmitted_features['image'],
                guide_vectors['image'],
                semantic_context=semantic_for_image,
                multiple_semantic_contexts=multiple_semantic_contexts,  # 传递语义上下文
                input_resolution=image_input_resolution,
                output_resolution=image_output_resolution,
                snr_db=snr_db
            )
            results['image_decoded'] = image_decoded
        
        # 视频解码（使用文本语义引导，如果提供）
        if 'video' in transmitted_features:
            if self.use_text_guidance_video and semantic_context is None:
                raise RuntimeError("use_text_guidance_video=True 但未提供 semantic_context。")
            semantic_for_video = semantic_context if self.use_text_guidance_video else None
            video_decoded = self.video_decoder(
                transmitted_features['video'],
                guide_vectors['video'],
                semantic_context=semantic_for_video,  # None 时视频解码不会走语义 cross-attention
                multiple_semantic_contexts=multiple_semantic_contexts
            )
            results['video_decoded'] = video_decoded
            if getattr(self.video_decoder, "last_semantic_gate_stats", None):
                stats = self.video_decoder.last_semantic_gate_stats
                results["video_semantic_gate_mean"] = stats.get("mean")
                results["video_semantic_gate_std"] = stats.get("std")
        
        return results
    
    def set_snr(self, snr_db: float):
        """设置信噪比"""
        self.channel.set_snr(snr_db)

    def set_bandwidth_ratio(self, ratio: float):
        """设置带宽门控比例。"""
        self.bandwidth_ratio = float(ratio)
        self.bandwidth_mask.set_ratio(self.bandwidth_ratio)
        self.conditional_bandwidth_gate.set_ratio(self.bandwidth_ratio)
    
    def reset_hidden_states(self):
        """重置所有隐藏状态"""
        if hasattr(self.video_encoder, 'reset_hidden_state'):
            self.video_encoder.reset_hidden_state()
        if hasattr(self.video_decoder, 'reset_hidden_state'):
            self.video_decoder.reset_hidden_state()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
        }


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


class DepthOnlyMultimodalJSCC(nn.Module):
    """Depth/Image/Video 三分支 JSCC（无文本接口）。"""

    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 4,
        img_embed_dims: List[int] = [96, 192, 384, 768],
        img_depths: List[int] = [2, 2, 6, 2],
        img_num_heads: List[int] = [3, 6, 12, 24],
        img_output_dim: int = 256,
        depth_output_dim: int = 128,
        video_hidden_dim: int = 256,
        video_num_frames: int = 5,
        video_output_dim: int = 256,
        channel_type: str = "awgn",
        snr_db: float = 10.0,
        power_normalization: bool = True,
    ):
        super().__init__()
        self.image_encoder = ImageJSCCEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dims=img_embed_dims,
            depths=img_depths,
            num_heads=img_num_heads,
            output_dim=img_output_dim,
        )
        self.image_decoder = ImageJSCCDecoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dims=img_embed_dims,
            depths=img_depths,
            num_heads=img_num_heads,
            input_dim=img_output_dim,
            semantic_context_dim=depth_output_dim,
        )
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
        self.channel = Channel(channel_type=channel_type, snr_db=snr_db, power_normalization=power_normalization)

        self.power_normalizer = nn.ModuleDict()
        if power_normalization:
            self.power_normalizer.update({
                'image': nn.LayerNorm(img_output_dim),
                'video': nn.LayerNorm(video_output_dim),
                'depth': nn.LayerNorm(depth_output_dim),
            })

    def _norm(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        if modality not in self.power_normalizer:
            return x
        if modality == 'video':
            v = x.permute(0, 1, 3, 4, 2)
            return self.power_normalizer[modality](v).permute(0, 1, 4, 2, 3)
        if x.dim() == 4:
            v = x.permute(0, 2, 3, 1)
            return self.power_normalizer[modality](v).permute(0, 3, 1, 2)
        return self.power_normalizer[modality](x)

    def forward(
        self,
        image_input: Optional[torch.Tensor] = None,
        depth_input: Optional[torch.Tensor] = None,
        video_input: Optional[torch.Tensor] = None,
        snr_db: Optional[float] = None,
    ) -> Dict[str, Any]:
        if snr_db is not None:
            self.channel.set_snr(snr_db)

        encoded, guides, out = {}, {}, {}

        if image_input is not None:
            image_encoded, image_guide = self.image_encoder(image_input)
            encoded['image'], guides['image'] = image_encoded, image_guide
        if depth_input is not None:
            depth_encoded, depth_guide = self.depth_encoder(depth_input)
            encoded['depth'], guides['depth'] = depth_encoded, depth_guide
        if video_input is not None:
            video_encoded, video_guide = self.video_encoder(video_input)
            encoded['video'], guides['video'] = video_encoded, video_guide

        transmitted = {}
        for m, feat in encoded.items():
            normed = self._norm(m, feat)
            transmitted[m] = self.channel(normed)
            out[f'{m}_encoded'] = feat
            out[f'{m}_transmitted'] = transmitted[m]

        if 'image' in transmitted:
            out['image_decoded'] = self.image_decoder(
                transmitted['image'],
                guides['image'],
                semantic_context=encoded.get('depth', None),
                input_resolution=getattr(self.image_encoder, 'last_latent_resolution', None),
                output_resolution=getattr(self.image_encoder, 'last_patch_resolution', None),
                output_size=getattr(self.image_encoder, 'last_input_size', None),
            )
        if 'depth' in transmitted:
            out['depth_decoded'] = self.depth_decoder(transmitted['depth'], guides['depth'])
        if 'video' in transmitted:
            out['video_decoded'] = self.video_decoder(
                transmitted['video'],
                guides['video'],
                semantic_context=encoded.get('depth', None),
                output_size=getattr(self.video_encoder, 'last_input_size', None),
            )

        out['rate_stats'] = {f'{k}_energy_rate_proxy': v.pow(2).mean() for k, v in encoded.items()}
        return out


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
            semantic_context=depth_latent_rx,
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
