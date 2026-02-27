# 严格审查版（v2）实现报告：模型减载功能量化与可控性改造

## 1. 修改目的（与审查要求逐条对齐）
本次改造仅围绕“可证据化”的两类能力：
1. **量化性增强**：把原有 `rate_stats` 中的二阶矩能量代理与离散码率代理拆分记录，避免统计口径混淆。
2. **可控性增强**：将纯前缀截断的带宽控制升级为“信道条件化门控（SNR + ratio）+ 可选稀疏化”的可控机制。

本报告不使用“彻底解决/完美控制”等绝对断言；不引入无法核验的链接；仅描述代码中可直接验证的行为。

---

## 2. 修改内容

### 2.1 条件带宽控制（Conditional Bandwidth Gate）
在 `multimodal_jscc.py` 中新增 `ConditionalBandwidthGate`，核心形式为：
\[
\tilde z = \gamma(c_{ch}) \odot z,\quad c_{ch}=[\text{SNR},\rho]
\]
其中 `gamma(c_ch)` 由 MLP 输出并经 `sigmoid` 限幅，随后按 `ratio` 执行可选前缀掩码。

实现位置：
- 新增模块：`ConditionalBandwidthGate`。
- 接入主模型：`MultimodalJSCC.forward()` 视频分支中优先调用条件门控；保留老 `BandwidthMask` 作为回退。
- `set_bandwidth_ratio()` 同步更新条件门控的 ratio。

### 2.2 VQ 瓶颈与经验熵统计
在 `multimodal_jscc.py` 中新增 `VectorQuantizer`（可选启用），支持：
- 输出离散索引 `video_vq_indices`；
- 输出 `video_vq_loss`；
- 输出经验熵 `video_empirical_entropy_bits`（单位 bit/element）。

统计公式：
\[
H(\hat z)= -\sum_i p_i\log_2 p_i,
\]
并新增码本上界日志：
\[
H(\hat z) \le \log_2 K.
\]

### 2.3 rate_stats 口径拆分
将主模型与 DepthOnly 分支中原 `v.pow(2).mean()` 统一命名为：
- `*_energy_rate_proxy`（明确是能量代理，不是离散码流比特）。

若启用 VQ，额外记录：
- `video_empirical_entropy_bits`；
- `video_vq_upper_bound_bits`。

### 2.4 损失端对“bits优先”的聚合策略
在 `losses.py` 的 `MultimodalLoss.forward()` 中更新 rate 正则聚合逻辑：
1. 优先聚合 `_bits` / `bpe` 项；
2. 若不存在，再回退到 `energy_rate_proxy`。

这样可避免把“熵代理”与“能量代理”无差别混加。

---

## 3. 对模型影响（可验证层面）

1. **训练日志可解释性提升**：
   现在可同时看到 `energy_rate_proxy` 与 `empirical_entropy_bits`，便于区分“功率约束”与“离散熵代理”。
2. **带宽控制可控性增强**：
   在同一 `ratio` 下，门控会随 SNR 变化，具备信道条件化调节能力（而不再仅仅是固定前缀截断）。
3. **优化目标更一致**：
   若存在 bits 统计，loss 端会优先约束 bits 项；仅在缺失时才退化到能量代理。
4. **边界声明**：
   本次改造仍属于“代理式码率建模”，未引入可部署算术编码器/物理链路全栈实现，因此不宣称真实吞吐已被端到端精确计量。

---

## 4. 数学与论文依据（真实可访问链接）

### 4.1 标准数学定义
- Shannon 熵（离散）：\(H(Z)=-\sum_i p_i\log_2 p_i\)
- 条件期望失真：\(D=\mathbb E[d(X,\hat X)]\)
- 变分目标（ELBO）标准形式：
\[
\max_\theta\;\mathbb E_q[\log p_\theta(x|\hat y,c)]-D_{KL}(q_\phi(z|x,\hat y)\|p_\theta(z|\hat y,c)).
\]

### 4.2 本轮保留论文链接（按审查清单）
1. Channel-Aware Vector Quantization for Robust Semantic Communication on Discrete Channels.  
   https://arxiv.org/abs/2510.18604
2. Extended Universal Joint Source-Channel Coding for Digital Semantic Communications: Improving Channel-Adaptability.  
   https://arxiv.org/abs/2602.14018
3. Zero-Shot Semantic Communication with Multimodal Foundation Models.  
   https://arxiv.org/abs/2502.18200
4. Joint Source-Channel-Generation Coding: From Distortion-oriented Reconstruction to Semantic-consistent Generation.  
   https://arxiv.org/abs/2601.12808
5. Timeliness-Aware Joint Source and Channel Coding for Adaptive Image Transmission.  
   https://arxiv.org/abs/2509.19754

---

## 5. 代码级注释与说明落实
本次对新增逻辑均加入了中文注释，重点包括：
- 条件门控为何引入（可控性）；
- VQ 熵统计为何引入（可量化性）；
- `energy_rate_proxy` 与 `bits` 的口径区别；
- loss 端“bits优先”聚合策略的目的。
