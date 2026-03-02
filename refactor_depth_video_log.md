# Depth+Video 专项重构日志

## 变更内容
1. **删除 Text/RGB 相关模块文件**
   - 已删除：`text_encoder.py`、`image_encoder.py`、`generative_image_decoder.py`、`generator_adapter.py`。
   - 原因：任务要求代码库仅保留 Depth 与 Video 两模态路径，避免无关分支被误调用。

2. **`multimodal_jscc.py` 精简**
   - 移除类：`MultimodalJSCC`、`DepthOnlyMultimodalJSCC`。
   - 移除 Text/Image/生成式图像相关 import。
   - 保留类：`BandwidthMask`、`ConditionalBandwidthGate`、`VectorQuantizer`、`DepthJSCCEncoder`、`DepthJSCCDecoder`、`JointLatentFusion`、`JointEntropyModel`、`MineEstimator`、`DepthVideoJSCC`。
   - 原因：保证仅有 Depth+Video 联合 JSCC 主路径，避免多模态混杂。

3. **`losses.py` 精简为五类核心损失**
   - 删除：`TextImageContrastiveLoss`、`TextLoss`、`ImageLoss`、`GenerativeImageLoss`、`DepthMultimodalLoss`、`MultimodalLoss`、`MSSSIM` 及 `_ssim/_gaussian_window`。
   - 保留：`VideoLoss`、`DepthLoss`、`DepthVideoLoss`、`PerceptualLoss`、`AdversarialLoss`。
   - 原因：仅保留 Depth+Video 训练所需目标，同时遵循“VideoLoss 已移除 MS-SSIM 节省显存”的既有设计。

4. **`data_loader.py` 重构为 Depth+Video 采样**
   - 删除 `_tokenize`、`_load_image`。
   - `__getitem__` 中移除 text/image 读取与异常处理分支。
   - `collate_multimodal_batch` 中移除 text/image pad 与拼接逻辑。
   - 默认 `required_modalities` 统一为 `("video", "depth")`。
   - 原因：数据管线与模型输入严格对齐，降低 batch 组装复杂度。

5. **`train.py` 重构为 Depth+Video 训练入口**
   - 删除 text/image 相关 CLI：文本引导图像、图像解码器类型、生成器路径与 gamma 参数等。
   - 删除 text/image 指标与日志跟踪项。
   - `create_loss_fn` 保持直接返回 `DepthVideoLoss`。
   - 原因：训练脚本与目标模态一致，避免遗留参数造成配置歧义。

6. **兼容性修复**
   - `video_encoder.py` 内联 `SNRModulator`（原定义位于已删除 `image_encoder.py`）。
   - 更新 `__init__.py` 导出项为 Depth+Video 相关类。
   - 原因：在删除 RGB 模块后，保证 Video 路径不因跨文件依赖断裂。

## 程序/数学/论文依据

### A. 端到端 JSCC 与深度学习语义通信
- Bourtsoulatze et al., *Deep Joint Source-Channel Coding for Wireless Image Transmission*, IEEE TCCN 2019.
  - https://ieeexplore.ieee.org/document/8736016
- Kurka & Gündüz, *DeepJSCC-f: Deep Joint Source-Channel Coding of Images with Feedback*, IEEE JSAC 2020.
  - https://ieeexplore.ieee.org/document/9149364

### B. Video 感知损失与对抗训练保留依据
- Zhang et al., *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)*, CVPR 2018.
  - https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html
- Mao et al., *Least Squares GAN (LSGAN)*, ICCV 2017.
  - https://openaccess.thecvf.com/content_iccv_2017/html/Mao_Least_Squares_Generative_ICCV_2017_paper.html

### C. 信息瓶颈/互信息修正项（与 `DepthVideoLoss` 中 KL/MI 正则一致）
- Tishby et al., *The Information Bottleneck Method*, 2000.
  - https://arxiv.org/abs/physics/0004057
- Belghazi et al., *Mutual Information Neural Estimation (MINE)*, ICML 2018.
  - https://proceedings.mlr.press/v80/belghazi18a.html

### D. SSIM/MS-SSIM 参考（说明为何可按现有 VideoLoss 注释移除）
- Wang et al., *Image Quality Assessment: From Error Visibility to Structural Similarity*, IEEE TIP 2004.
  - https://ieeexplore.ieee.org/document/1284395
- Wang et al., *Multi-Scale Structural Similarity for Image Quality Assessment*, Asilomar 2003.
  - https://ieeexplore.ieee.org/document/1292216

> 说明：本次日志仅引用公开可访问文献/论文主页；未添加无法验证来源的链接。

## 二次清理（按审阅意见追加）

7. **`config.py` 移除 Text/RGB 配置残留**
   - 在 `TrainingConfig` / `EvaluationConfig` 中删除文本词表、文本编码器、文本引导、Swin 图像分支、VAE 生成器等字段。
   - 保留与 depth+video 路径直接相关的参数（DepthVideoJSCC、视频时序/熵模型、SNR/信道、OMIB/MINE、训练调度）。
   - 原因：配置层面应与计算图一致，避免 checkpoint 和运行参数继续携带废弃模态。

8. **`train.py` 清除 checkpoint 与判别器中的 RGB 参数残留**
   - 精简 `model_config`，移除 `img_embed_dims/img_depths/img_num_heads/generator_*` 等历史字段。
   - 判别器实例化不再显式传入 `image_input_nc=3`。
   - 原因：训练态元信息应仅表达 depth+video 体系，防止恢复训练时加载无关配置。

9. **`cross_attention.py` 废弃三模态交互实现，仅保留基础 CrossAttention**
   - 删除 `MultiModalCrossAttention` 及 text↔image↔video 两两耦合结构，仅保留 video 语义引导实际使用的标准缩放点积交叉注意力。
   - 原因：避免保留未使用的三模态死代码，同时不破坏视频分支对 CrossAttention 的依赖。

10. **`data_loader.py` 清除初始化参数残留**
   - 从 `MultimodalDataset` / `MultimodalDataLoader` 的 `__init__` 去掉 `text_tokenizer`、`max_text_length`。
   - 移除依赖 `item["text"]` 的版本检测逻辑。
   - 原因：数据接口层面严格限定到 depth/video，避免调用方误传文本相关参数。

## 新增清理依据（程序/论文/数学）
- Scaled Dot-Product Attention（交叉注意力基础）: Vaswani et al., 2017.
  - https://arxiv.org/abs/1706.03762
- Swin Transformer（已清理的图像分支历史配置来源）: Liu et al., 2021.
  - https://arxiv.org/abs/2103.14030
- Latent Diffusion / SD-VAE（已清理的 generator 配置来源）: Rombach et al., 2022.
  - https://arxiv.org/abs/2112.10752
