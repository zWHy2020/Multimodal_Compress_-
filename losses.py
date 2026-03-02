"""
Depth + Video JSCC losses.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("警告: LPIPS库未安装，将使用VGG特征损失作为替代。建议安装: pip install lpips")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class PerceptualLoss(nn.Module):
    """
    感知损失（LPIPS或VGG特征损失）
    
    使用LPIPS（如果可用）或VGG特征损失来提升重建的感知质量。
    """
    def __init__(self, use_lpips: bool = True):
        super().__init__()
        self.use_lpips = use_lpips and LPIPS_AVAILABLE
        
        if self.use_lpips:
            # 使用LPIPS（AlexNet backbone，轻量级）
            self.lpips_fn = lpips.LPIPS(net='alex', verbose=False)
            # 冻结LPIPS参数
            for param in self.lpips_fn.parameters():
                param.requires_grad = False
        else:
            # 使用VGG特征损失作为替代
            vgg = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
            )
            # 加载预训练权重（如果可用）
            try:
                # 尝试加载预训练VGG权重
                vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
                vgg[0].weight.data = vgg16.features[0].weight.data
                vgg[1].weight.data = vgg16.features[2].weight.data
                vgg[3].weight.data = vgg16.features[5].weight.data
                vgg[4].weight.data = vgg16.features[7].weight.data
            except:
                pass
            self.vgg = vgg
            # 冻结VGG参数
            for param in self.vgg.parameters():
                param.requires_grad = False
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算感知损失
        
        Args:
            pred (torch.Tensor): 预测图像 [B, C, H, W]，范围[0, 1]
            target (torch.Tensor): 目标图像 [B, C, H, W]，范围[0, 1]
            
        Returns:
            torch.Tensor: 感知损失值
        """
        if self.use_lpips:
            # LPIPS需要输入范围[-1, 1]
            pred_norm = pred * 2.0 - 1.0
            target_norm = target * 2.0 - 1.0
            loss = self.lpips_fn(pred_norm, target_norm).mean()
        else:
            # VGG特征损失
            pred_features = self.vgg(pred)
            target_features = self.vgg(target)
            loss = F.mse_loss(pred_features, target_features)
        
        return loss


class VideoLoss(nn.Module):
    """
    视频损失 (轻量级优化版)
    
    只使用L1重建损失和简单的时序损失，移除MS-SSIM以节省显存。
    """
    def __init__(
        self, 
        reconstruction_weight: float = 1.0, 
        perceptual_weight: float = 0.0,  # 默认禁用
        temporal_weight: float = 0.1,
        temporal_consistency_weight: float = 0.0,
        temporal_perceptual_weight: float = 0.0,
        color_consistency_weight: float = 0.0,
        data_range: float = 1.0,
        normalize: bool = False,

    ):
        super().__init__()
        self.recon_weight = reconstruction_weight
        self.percep_weight = perceptual_weight
        self.temp_weight = temporal_weight
        self.consistency_weight = temporal_consistency_weight
        self.temporal_perceptual_weight = temporal_perceptual_weight
        self.color_consistency_weight = color_consistency_weight
        self.data_range = data_range
        self.normalize = normalize
        #self.epsilon = 1e-6
        
        # 重建损失 (L1)
        self.recon_loss_fn = nn.L1Loss()
        
        # 感知损失（LPIPS或VGG）
        if self.percep_weight > 0:
            self.percep_loss_fn = PerceptualLoss(use_lpips=LPIPS_AVAILABLE)
        else:
            self.percep_loss_fn = None
        self.register_buffer("imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def _maybe_denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return tensor
        return tensor * self.imagenet_std + self.imagenet_mean
    
    #def _normalize_to_range(self, img: torch.Tensor) -> torch.Tensor:
        """
        将图像归一化到 [0, 1] 范围
        
        如果输入使用了 ImageNet 归一化（范围约在 [-2.5, 2.5]），
        需要先反归一化，然后归一化到 [0, 1]。
        """
        # 检测是否使用了 ImageNet 归一化（通过数据范围判断）
        #img_min = img.min().item()
        #img_max = img.max().item()
        
        # 如果数据范围明显超出 [0, 1]，可能是 ImageNet 归一化的结果
        #if img_min < -0.5 or img_max > 1.5:
            # ImageNet 归一化参数
            #imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=img.device, dtype=img.dtype)
            #imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=img.device, dtype=img.dtype)
            
            # 添加维度以便广播 [1, 3, 1, 1]
            #mean = imagenet_mean.view(1, 3, 1, 1)
            #std = imagenet_std.view(1, 3, 1, 1)
            
            # 反归一化: x = normalized * std + mean
            #img = img * std + mean
        
        # 确保在 [0, 1] 范围内
        #img = torch.clamp(img, 0.0, 1.0)
        
        #return img
            
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, T, C, H, W = pred.shape
        
        pred_f32 = self._maybe_denormalize(pred.float())
        target_f32 = self._maybe_denormalize(target.float())
        loss_recon_avg = F.l1_loss(pred_f32, target_f32)
        loss_temp = torch.tensor(0.0, device=pred.device)
        loss_consistency = torch.tensor(0.0, device=pred.device)
        loss_percep = torch.tensor(0.0, device=pred.device)
        loss_temporal_percep = torch.tensor(0.0, device=pred.device)
        loss_color = torch.tensor(0.0, device=pred.device)
        if pred_f32.size(1) > 1 and self.temp_weight > 0:
            pred_diff = pred_f32[:, 1:] - pred_f32[:, :-1]
            target_diff = target_f32[:, 1:] - target_f32[:, :-1]
            loss_temp = F.l1_loss(pred_diff, target_diff)
        if pred_f32.size(1) > 1 and self.consistency_weight > 0:
            pred_diff = pred_f32[:, 1:] - pred_f32[:, :-1]
            loss_consistency = pred_diff.abs().mean()
        if self.percep_loss_fn is not None and self.percep_weight > 0:
            try:
                pred_frames = pred_f32.reshape(B * T, C, H, W)
                target_frames = target_f32.reshape(B * T, C, H, W)
                loss_percep = self.percep_loss_fn(pred_frames, target_frames)
            except Exception:
                loss_percep = torch.tensor(0.0, device=pred.device)
        if pred_f32.size(1) > 1 and self.temporal_perceptual_weight > 0 and self.percep_loss_fn is not None:
            try:
                pred_diff = pred_f32[:, 1:] - pred_f32[:, :-1]
                target_diff = target_f32[:, 1:] - target_f32[:, :-1]
                pred_frames = pred_diff.reshape(B * (T - 1), C, H, W)
                target_frames = target_diff.reshape(B * (T - 1), C, H, W)
                loss_temporal_percep = self.percep_loss_fn(pred_frames, target_frames)
            except Exception:
                loss_temporal_percep = torch.tensor(0.0, device=pred.device)
        if self.color_consistency_weight > 0:
            pred_mean = pred_f32.mean(dim=(-2, -1), keepdim=True)
            target_mean = target_f32.mean(dim=(-2, -1), keepdim=True)
            pred_std = pred_f32.std(dim=(-2, -1), keepdim=True)
            target_std = target_f32.std(dim=(-2, -1), keepdim=True)
            loss_color = F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)
        total_loss = (
            (self.recon_weight * loss_recon_avg) +
            (self.temp_weight * loss_temp) +
            (self.consistency_weight * loss_consistency) +
            (self.percep_weight * loss_percep) +
            (self.temporal_perceptual_weight * loss_temporal_percep) +
            (self.color_consistency_weight * loss_color)
        )
        return total_loss, {
            'video_recon_loss_l1': loss_recon_avg.item(),
            'video_temporal_loss_l1': loss_temp.item(),
            'video_temporal_consistency_loss': loss_consistency.item(),
            'video_percep_loss': loss_percep.item(),
            'video_temporal_percep_loss': loss_temporal_percep.item(),
            'video_color_consistency_loss': loss_color.item(),
        }
        #diff = pred -target
        #loss_recon_avg= torch.mean(torch.sqrt(diff * diff + self.epsilon))
        #loss_percep_avg = torch.tensor(0.0, device=pred.device)
        #loss_temp = torch.tensor(0.0, device=pred.device)
        #if T > 1 and self.temp_weight > 0:
            #pred_diff = pred[:, 1:] - pred[:, :-1]
            #target_diff = target[:, 1:] - target[:, :-1]
            #diff_motion = pred_diff - target_diff
            #loss_temp = torch.mean(torch.sqrt(diff_motion * diff_motion + self.epsilon))

        
        # 1. 计算逐帧的重建损失 (L1) - 已移除MS-SSIM以节省显存
        #for t in range(T):
            #frame_pred = pred_normalized[:, t]
            #frame_target = target_normalized[:, t]
            
            # L1 重建损失
            #loss_recon_total += self.recon_loss_fn(frame_pred, frame_target)
        
        #loss_recon_avg = loss_recon_total / T
        #loss_percep_avg = torch.tensor(0.0, device=pred.device)  # 感知损失已移除
        
        # 2. 计算时序损失 (帧间差异, L1)
        #loss_temp = torch.tensor(0.0, device=pred.device)
        #if T > 1 and self.temp_weight > 0:
            #pred_diff = pred_normalized[:, 1:] - pred_normalized[:, :-1]
            #target_diff = target_normalized[:, 1:] - target_normalized[:, :-1]
            #loss_temp = self.recon_loss_fn(pred_diff, target_diff) # 使用 L1 计算运动差异
        
        # 3. 汇总总损失（添加数值稳定性检查）
        # 检查损失值是否有效
        #if torch.isnan(loss_recon_avg) or torch.isinf(loss_recon_avg):
            #loss_recon_avg = torch.tensor(0.0, device=pred.device)
        #if torch.isnan(loss_percep_avg) or torch.isinf(loss_percep_avg):
            #loss_percep_avg = torch.tensor(0.0, device=pred.device)
        #if torch.isnan(loss_temp) or torch.isinf(loss_temp):
            #loss_temp = torch.tensor(0.0, device=pred.device)
        
        #total_loss = (
            #(self.recon_weight * loss_recon_avg) +
            #(self.percep_weight * loss_percep_avg) +
            #(self.temp_weight * loss_temp)
        #)
        
        # 最终检查
        #if torch.isnan(total_loss) or torch.isinf(total_loss):
            #total_loss = torch.tensor(0.0, device=pred.device)
        
       # return total_loss, {
            #'video_recon_loss_l1': loss_recon_avg.item() if not torch.isnan(loss_recon_avg) else 0.0,
            ##'video_percep_loss_msssim': 0.0,  # 已移除
            #'video_temporal_loss_l1': loss_temp.item() if not torch.isnan(loss_temp) else 0.0
        #}


# --------------------------------------------------------------------------
# 对抗损失（Phase 3: GAN Loss）
# --------------------------------------------------------------------------


class AdversarialLoss(nn.Module):
    """
    对抗损失（Phase 3: GAN Loss）
    
    使用最小二乘GAN损失（LSGAN），比标准GAN损失更稳定。
    """
    def __init__(self, target_real_label: float = 1.0, target_fake_label: float = 0.0):
        super().__init__()
        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label
        self.loss_fn = nn.MSELoss()
    
    def forward(
        self,
        discriminator_output: torch.Tensor,
        target_is_real: bool
    ) -> torch.Tensor:
        """
        计算对抗损失
        
        Args:
            discriminator_output: 判别器输出 [B, 1, H, W] 或 [B, 1]
            target_is_real: 目标是否为真实图像
            
        Returns:
            torch.Tensor: 对抗损失值
        """
        if target_is_real:
            target = torch.full_like(discriminator_output, self.target_real_label)
        else:
            target = torch.full_like(discriminator_output, self.target_fake_label)
        
        loss = self.loss_fn(discriminator_output, target)
        return loss


# --------------------------------------------------------------------------
# 多模态损失协调器 (保持不变，仅更新调用的子损失)
# --------------------------------------------------------------------------


class DepthLoss(nn.Module):
    """深度重建损失：L1 + 边缘一致性。"""

    def __init__(self, l1_weight: float = 1.0, edge_weight: float = 0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.edge_weight = edge_weight

    @staticmethod
    def _gradient_map(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        return gx, gy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        l1 = F.l1_loss(pred, target)
        gx_p, gy_p = self._gradient_map(pred)
        gx_t, gy_t = self._gradient_map(target)
        edge = F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)
        total = self.l1_weight * l1 + self.edge_weight * edge
        return total, {'depth_l1_loss': l1.item(), 'depth_edge_loss': edge.item()}


class DepthVideoLoss(nn.Module):
    """深度图+视频双模态损失（无文本/图像项）。"""

    def __init__(
        self,
        depth_weight: float = 1.0,
        video_weight: float = 1.0,
        rate_weight: float = 1e-4,
        use_omib_like: bool = True,
        ib_beta: float = 1e-4,
        ib_beta_min: float = 0.0,
        ib_beta_max: Optional[float] = None,
        omib_eps: float = 1e-8,
        mi_correction_weight: float = 1.0,
    ):
        super().__init__()
        self.depth_weight = depth_weight
        self.video_weight = video_weight
        self.rate_weight = rate_weight
        self.use_omib_like = bool(use_omib_like)
        self.ib_beta = float(ib_beta)
        self.ib_beta_min = float(ib_beta_min)
        self.ib_beta_max = None if ib_beta_max is None else float(ib_beta_max)
        self.omib_eps = float(omib_eps)
        self.mi_correction_weight = float(mi_correction_weight)
        self.depth_loss_fn = DepthLoss()
        self.video_loss_fn = VideoLoss()

    def _compute_dynamic_r(self, depth_task_loss: torch.Tensor, video_task_loss: torch.Tensor) -> torch.Tensor:
        ratio = (video_task_loss.detach().clamp_min(self.omib_eps) / depth_task_loss.detach().clamp_min(self.omib_eps))
        r = 1.0 - torch.tanh(torch.log(ratio))
        return r.clamp(0.0, 2.0)

    @staticmethod
    def _modality_rate_weights(r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 令两模态平均权重保持在 1 附近，避免整体 rate loss 尺度突变。
        denom = (1.0 + r).clamp_min(1e-8)
        lambda_depth = 2.0 / denom
        lambda_video = 2.0 * r / denom
        return lambda_depth, lambda_video

    def _effective_ib_beta(self) -> float:
        beta = max(self.ib_beta, self.ib_beta_min)
        if self.ib_beta_max is not None:
            beta = min(beta, self.ib_beta_max)
        return beta

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        device = None
        for v in predictions.values():
            if isinstance(v, torch.Tensor):
                device = v.device
                break
        if device is None:
            device = torch.device('cpu')

        total = torch.tensor(0.0, device=device)
        loss_dict: Dict[str, float] = {}

        depth_task_loss = torch.tensor(0.0, device=device)
        video_task_loss = torch.tensor(0.0, device=device)

        if 'depth_decoded' in predictions and 'depth' in targets:
            depth_loss, depth_comps = self.depth_loss_fn(predictions['depth_decoded'], targets['depth'])
            depth_loss = self.depth_weight * depth_loss
            total = total + depth_loss
            depth_task_loss = depth_loss.detach()
            loss_dict['depth_loss'] = depth_loss.item()
            loss_dict.update(depth_comps)

        if 'video_decoded' in predictions and 'video' in targets:
            video_loss, video_comps = self.video_loss_fn(predictions['video_decoded'], targets['video'])
            video_loss = self.video_weight * video_loss
            total = total + video_loss
            video_task_loss = video_loss.detach()
            loss_dict['video_loss'] = video_loss.item()
            loss_dict.update(video_comps)

        if 'rate_stats' in predictions and predictions['rate_stats']:
            rate_stats = predictions['rate_stats']
            r = self._compute_dynamic_r(depth_task_loss, video_task_loss)
            lambda_depth, lambda_video = self._modality_rate_weights(r)

            # 优先使用分项码率；若不存在则回退到统一平均。
            if all(k in rate_stats for k in ('shared_bpe', 'depth_private_bpe', 'video_private_bpe')):
                rate_penalty = (
                    rate_stats['shared_bpe']
                    + lambda_depth * rate_stats['depth_private_bpe']
                    + lambda_video * rate_stats['video_private_bpe']
                )
                mi_correction = None
                if 'cross_modal_mi_bits' in rate_stats:
                    mi_correction = self.mi_correction_weight * rate_stats['cross_modal_mi_bits']
                    rate_penalty = rate_penalty - mi_correction
            else:
                rate_penalty = sum(v for v in rate_stats.values()) / len(rate_stats)
                mi_correction = None

            rate_loss = self.rate_weight * rate_penalty
            total = total + rate_loss
            loss_dict['rate_loss'] = rate_loss.item()
            loss_dict['omib_dynamic_r'] = r.item()
            loss_dict['lambda_depth'] = lambda_depth.item()
            loss_dict['lambda_video'] = lambda_video.item()
            if mi_correction is not None:
                loss_dict['cross_modal_mi_bits'] = rate_stats['cross_modal_mi_bits'].item()
                loss_dict['mi_correction'] = mi_correction.item()

        if self.use_omib_like and 'omib_stats' in predictions:
            omib_stats = predictions['omib_stats']
            if 'depth_kl' in omib_stats and 'video_kl' in omib_stats:
                r = self._compute_dynamic_r(depth_task_loss, video_task_loss)
                beta_eff = self._effective_ib_beta()
                omib_kl = omib_stats['depth_kl'] + r * omib_stats['video_kl']
                omib_loss = beta_eff * omib_kl
                total = total + omib_loss
                loss_dict['omib_like_loss'] = omib_loss.item()
                loss_dict['omib_depth_kl'] = omib_stats['depth_kl'].item()
                loss_dict['omib_video_kl'] = omib_stats['video_kl'].item()
                loss_dict['omib_beta_eff'] = float(beta_eff)

        loss_dict['total_loss'] = total
        return loss_dict

