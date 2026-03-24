"""Depth+Video JSCC 配置模块。"""

import os
from datetime import datetime


class TrainingConfig:
    """训练配置类（仅 Depth+Video）。"""

    def __init__(self):
        self.seed = 42
        self.device = None
        self.num_workers = 0
        self.pin_memory = True

        self.workdir = './checkpoints'
        self.log_dir = './logs'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f'multimodal_jscc_{timestamp}'
        self.save_dir = os.path.join(self.workdir, self.run_name)
        self.log_file = os.path.join(self.log_dir, f'{self.run_name}.log')

        self.num_epochs = 100
        self.batch_size = 32
        self.learning_rate = 5e-5
        self.weight_decay = 1e-4
        self.grad_clip_norm = 1.0
        self.gradient_accumulation_steps = 2
        self.use_amp = True
        self.use_gradient_checkpointing = True

        self.lr_scheduler = 'cosine'
        self.lr_step_size = 30
        self.lr_gamma = 0.1
        self.lr_min = 1e-6

        self.val_freq = 10
        self.save_freq = 20
        self.print_freq = 200

        # losses
        self.video_weight = 1.0
        self.depth_weight = 1.0
        self.reconstruction_weight = 1.0
        self.perceptual_weight = 0.01
        self.temporal_weight = 0.05
        self.rate_weight = 1e-4
        self.temporal_consistency_weight = 0.02

        # OMIB-like
        self.use_omib_like = True
        self.ib_beta = 1e-4
        self.ib_beta_min = 0.0
        self.ib_beta_max = None
        self.mi_correction_weight = 1.0
        self.use_mine_beta_bound = True
        self.mine_beta_estimate_steps = 20
        self.mine_train_steps = 50
        self.mine_hidden_dim = 128
        self.mine_lr = 1e-4

        # channel/SNR
        self.train_snr_min = -5.0
        self.train_snr_max = 15.0
        self.train_snr_random = False
        self.train_snr_strategy = "curriculum"
        self.snr_db = 10.0

        self.bandwidth_ratio_start = 1.0
        self.bandwidth_ratio_end = 0.75
        self.bandwidth_warmup_epochs = 15
        self.bandwidth_anneal_epochs = 50

        # model
        self.img_size = (256, 512)
        self.patch_size = 4
        self.video_hidden_dim = 192
        self.video_num_frames = 10
        self.video_use_optical_flow = True
        self.video_use_convlstm = True
        self.video_output_dim = 192
        self.depth_output_dim = 128
        self.shared_latent_dim = 128
        self.video_decoder_type = "swin"
        self.video_unet_base_channels = 64
        self.video_unet_num_down = 4
        self.video_unet_num_res_blocks = 3
        self.video_decode_chunk_size = None
        self.video_gop_size = 5
        self.video_latent_downsample_factor = 8
        self.video_latent_downsample_stride = self.video_latent_downsample_factor
        self.video_entropy_max_exact_quantile_elems = 2_000_000
        self.video_entropy_quantile_sample_size = 262_144
        self.channel_type = "awgn"
        self.model_mode = "joint"

        # data
        self.data_dir = None
        self.train_manifest = None
        self.val_manifest = None
        self.max_video_frames = 10
        self.max_depth_maps = 1
        self.video_clip_len = self.max_video_frames
        self.video_stride = 1
        self.video_sampling_strategy = "contiguous_clip"
        self.video_eval_sampling_strategy = "uniform"
        self.max_samples = 65536
        self.allow_missing_modalities = False
        self.strict_data_loading = True

        # adversarial
        self.use_adversarial = False
        self.gan_enable_epoch = 5
        self.d_updates_per_g = 1
        self.gan_weight = 0.01
        self.discriminator_weight = self.gan_weight
        self.use_r1_regularization = False
        self.r1_gamma = 10.0
        self.ddp_find_unused_parameters = False
        self.use_quantization_noise = True
        self.quantization_noise_range = 0.5
        self.normalize = True
        self.video_perceptual_weight = 0.01
        self.temporal_perceptual_weight = 0.02
        self.color_consistency_weight = 0.02

    def print_config(self, logger=None):
        log_func = logger.info if logger else print
        log_func("\n=== 当前生效的配置 (TrainingConfig) ===")
        log_func(f"Use Adversarial: {getattr(self, 'use_adversarial', False)}")
        log_func(
            "Video sampling: clip_len="
            f"{getattr(self, 'video_clip_len', None)} stride="
            f"{getattr(self, 'video_stride', None)} train_strategy="
            f"{getattr(self, 'video_sampling_strategy', None)} eval_strategy="
            f"{getattr(self, 'video_eval_sampling_strategy', None)}"
        )
        log_func(
            "Video GOP/latent: gop_size="
            f"{getattr(self, 'video_gop_size', None)} latent_factor="
            f"{getattr(self, 'video_latent_downsample_factor', getattr(self, 'video_latent_downsample_stride', None))}"
        )
        log_func("=========================================\n")


class EvaluationConfig:
    """评估配置类（仅 Depth+Video）。"""

    def __init__(self):
        self.seed = 42
        self.device = None
        self.num_workers = 4
        self.pin_memory = True

        self.log_dir = './logs'
        self.save_dir = './evaluation_results'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f'evaluation_{timestamp}'
        self.log_file = os.path.join(self.log_dir, f'{self.run_name}.log')
        self.result_dir = os.path.join(self.save_dir, self.run_name)

        self.model_path = None
        self.batch_size = 8
        self.test_num = None

        self.snr_list = [-5, -10, -5, 0, 5, 10, 15, 20]
        self.rate_list = None
        self.snr_db = 10.0
        self.snr_random = False
        self.snr_min = -5.0
        self.snr_max = 15.0

        self.use_patch_inference = False
        self.patch_size = 128
        self.patch_overlap = 32

        self.data_dir = None
        self.test_manifest = None

        self.save_images = False
        self.image_save_dir = None

        self.image_size = (256, 512)
        self.max_video_frames = 10
        self.max_depth_maps = 1
        self.video_clip_len = self.max_video_frames
        self.video_stride = 1
        self.video_sampling_strategy = "contiguous_clip"
        self.video_eval_sampling_strategy = "uniform"
        self.video_gop_size = 5
        self.infer_window_len = None
        self.infer_window_stride = None
        self.infer_window_blend = "uniform"
        self.max_output_frames = None

        self.img_size = (256, 512)
        self.img_patch_size = 4
        self.mlp_ratio = 4.0
        self.video_hidden_dim = 256
        self.video_num_frames = 10
        self.video_use_optical_flow = True
        self.video_use_convlstm = True
        self.video_output_dim = 256
        self.video_decoder_type = "swin"
        self.video_unet_base_channels = 64
        self.video_unet_num_down = 4
        self.video_unet_num_res_blocks = 3
        self.video_decode_chunk_size = None
        self.use_amp = False
        self.use_gradient_checkpointing = False

        self.channel_type = "awgn"
        self.normalize = True
        self.video_latent_downsample_factor = 8
        self.video_latent_downsample_stride = self.video_latent_downsample_factor
        self.video_entropy_max_exact_quantile_elems = 2_000_000
        self.video_entropy_quantile_sample_size = 262_144
