# 多模态实时感知联合压缩模型模块化报告

## 1. 模块化目标

围绕仓库目标“多模态实时感知数据联合压缩减载传输 + 信息丢失可控 + 重建质量保持”，本次模块化遵循以下约束：

1. **功能等价**：不改变原脚本的核心功能与计算路径，只做结构拆分与接口标准化。
2. **高内聚**：每个模块只负责一类能力（编码、融合、信道、熵建模、系统编排等）。
3. **低耦合**：模块间只通过明确 API/数据结构交互，避免跨文件共享隐式状态。
4. **可独立开发与测试**：任意模块可被单独导入、替换和单测。
5. **可组合扩展**：后续可替换视频编码器、熵模型、门控策略，不影响主系统调用方式。

---

## 2. 模块化后的标准 API 设计

新增统一接口定义文件：`modules/api.py`。

### 2.1 数据接口（TypedDict）

- `EncoderOutput`: 编码器输出约定（`latent`, `guide`）。
- `FusionOutput`: 融合输出约定（`shared_latent`, `depth_private`, `video_private` 等）。
- `RateStats`: 码率统计约定（`joint_bpe`, `shared_bpe` 等）。
- `ModelForwardOutput`: 主模型 `forward` 输出约定，统一训练/推理可见字段。

### 2.2 调用接口（前向约定）

- `ModelForwardAPI.forward(depth_input, video_input, snr_db=None) -> ModelForwardOutput`
- 输入输出形状约束：
  - 深度输入：`[B, 1, H, W]`
  - 视频输入：`[B, T, 3, H, W]`
  - 输出包含重建结果、信道发送特征、熵统计、码率统计等

该接口保证了训练脚本、评估脚本仅依赖标准字段，不依赖内部实现细节。

---

## 3. 支撑模块化设计的数学依据

### 3.1 率失真与信息瓶颈目标

联合压缩核心是最小化：
\[
\mathcal{L} = \mathbb{E}[d(x, \hat{x})] + \lambda R
\]
其中失真项由深度/视频重建误差组成，码率由熵模型近似。该目标来自经典率失真理论。

### 3.2 联合潜变量分解（共享 + 私有）

`JointLatentFusion` 将跨模态表示分解为共享语义 `z_s` 与私有残差 `z_d, z_v`，等价于多视角表示学习中的公共因子分解思想：
\[
z_d = z_s^{(d)} + r_d,\quad z_v = z_s^{(v)} + r_v
\]
便于单独估计码率并约束跨模态冗余。

### 3.3 熵模型与码率估计

`JointEntropyModel` 用高斯先验近似每一路潜变量的负对数似然，换算 bit/element：
\[
R \approx \mathbb{E}[-\log_2 p(z)]
\]
在工程上对应可训练可微分的码率代理。

### 3.4 互信息校正（MINE）

`MineEstimator` 使用 Donsker–Varadhan 形式估计互信息下界：
\[
I(X;Y) \ge \mathbb{E}_{P_{XY}}[T] - \log \mathbb{E}_{P_XP_Y}[e^T]
\]
用于约束跨模态私有分量中的冗余信息。

### 3.5 条件带宽门控

`ConditionalBandwidthGate` 基于 `(SNR, ratio)` 生成调制系数并执行前缀稀疏化，实现“压缩过程中的信息丢失可控”。这属于条件特征调制与资源分配结合策略。

---

## 4. 论文与公开资料依据（可访问真实链接）

1. Shannon, C. E. (1948). *A Mathematical Theory of Communication*.  
   链接：https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf
2. Bourtsoulatze, E., Kurka, D. B., & Gündüz, D. (2019). *Deep Joint Source-Channel Coding for Wireless Image Transmission*.  
   链接：https://arxiv.org/abs/1809.01733
3. Ballé, J., Laparra, V., & Simoncelli, E. P. (2017). *End-to-end Optimized Image Compression*.  
   链接：https://arxiv.org/abs/1611.01704
4. van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). *Neural Discrete Representation Learning (VQ-VAE)*.  
   链接：https://arxiv.org/abs/1711.00937
5. Belghazi, M. I. et al. (2018). *MINE: Mutual Information Neural Estimation*.  
   链接：https://arxiv.org/abs/1801.04062
6. Perez, E. et al. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer*.  
   链接：https://arxiv.org/abs/1709.07871

---

## 5. 最终模块化后本地仓库模型结构

> 仅列出与模型核心相关的结构（保持原训练/推理入口不变）。

```text
Multimodal_Compress_-/
├── multimodal_jscc.py                  # 兼容导出层（保持旧导入路径可用）
├── channel.py                          # 信道模块（保留）
├── video_encoder.py                    # 视频编码器（保留）
├── video_unet.py                       # 视频解码器（保留）
├── train.py                            # 训练入口（无需改动）
├── inference.py                        # 推理入口（保留）
├── evaluate.py                         # 评估入口（保留）
└── modules/
    ├── __init__.py                     # 模块化统一导出
    ├── api.py                          # 标准接口定义（TypedDict/API约束）
    ├── gating.py                       # BandwidthMask / ConditionalBandwidthGate
    ├── quantization.py                 # VectorQuantizer
    ├── depth_codec.py                  # DepthJSCCEncoder / DepthJSCCDecoder
    ├── fusion.py                       # JointLatentFusion / JointEntropyModel / MineEstimator
    └── system.py                       # DepthVideoJSCC 系统编排
```

---

## 6. 各模块职责与解耦关系

- `modules/depth_codec.py`：只负责深度模态编码解码。
- `modules/gating.py`：只负责带宽控制策略。
- `modules/quantization.py`：只负责离散量化与熵统计。
- `modules/fusion.py`：只负责融合、码率代理、互信息估计。
- `modules/system.py`：只做“装配与流程编排”，不承载具体子模块算法细节。
- `multimodal_jscc.py`：仅做兼容导出，避免影响现有 `from multimodal_jscc import ...` 语句。

该拆分实现“算法模块可替换、训练脚本无感、接口稳定可复用”。
