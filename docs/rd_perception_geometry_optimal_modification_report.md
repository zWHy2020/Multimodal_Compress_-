# 多模态 JSCC（视频+深度）最优替代方案报告（RD+感知+几何）

## 1. 目标与约束

本报告面向仓库当前的双模态任务（视频 + 深度图）与联合信源信道编码（JSCC）传输目标，给出可直接落地到现有代码结构（`losses.py`, `metrics.py`, 训练/验证循环）的替代方案：

- **目标1（RD）**：在固定带宽/码率下最小化重建失真；
- **目标2（Perception）**：提升视觉主观质量与时序稳定性；
- **目标3（Geometry）**：保持深度的尺度关系与3D结构一致性；
- **目标4（Engineering）**：优先采用“低改造成本 + 高学术公认度 + 2022+时效性”的组合。

---

## 2. 从当前实现到最优组合的映射

### 2.1 当前实现（仓库事实）

- 视频损失：`L1 + 帧差L1(+可选2D LPIPS/VGG)`；
- 深度损失：`L1 + 2D梯度一致性`；
- 对抗损失：LSGAN(MSE)；
- 指标：PSNR/SSIM + 文本 BLEU/ROUGE。

### 2.2 最优替代目标（分层）

> 推荐使用三层耦合目标，而非“单一指标替换”。

1. **RD层（必备）**：
   - 视频：`Charbonnier/L1`；
   - 深度：`SILog` 主项 + 结构梯度项。
2. **感知层（必备）**：
   - 视频：3D时空特征损失（I3D/VideoMAE）；
   - （可选）Hinge GAN + R1 稳定项。
3. **几何层（深度必备）**：
   - 法向一致（VNL 或 surface-normal cosine）。

---

## 3. 损失函数的完整数学定义与推导

记训练样本为 \((V,D)\)，模型输出为 \((\hat V, \hat D)\)。

## 3.1 视频分支损失

### 3.1.1 RD主项：Charbonnier 像素损失

\[
L_{v,pix}=\frac{1}{N_v}\sum_i \rho(v_i-\hat v_i),\quad
\rho(x)=\sqrt{x^2+\epsilon^2}
\]

- \(N_v\)：视频像素总数（含时序维）；
- \(\epsilon\)：平滑常数（如 \(10^{-3}\)），防止梯度在0附近不稳定。

**作用**：相较纯L2对异常值更稳健，相较纯L1更平滑，适合JSCC噪声扰动。

### 3.1.2 时空感知项：3D backbone feature loss

令 \(\phi_{3D}(\cdot)\) 为冻结的 I3D/VideoMAE 特征提取器：

\[
L_{v,st}=\sum_{l\in\mathcal L} w_l\,\|\phi^l_{3D}(V)-\phi^l_{3D}(\hat V)\|_2^2
\]

- \(\mathcal L\)：选取的层集合；
- \(w_l\)：第 \(l\) 层权重。

**作用**：显式约束时空语义一致，抑制逐帧2D感知导致的 flicker。

### 3.1.3 对抗项（可选高保真模式）

判别器采用多帧输入（3D Patch 判别器）时，Hinge 目标：

\[
L_D=\mathbb E[\max(0,1-D(V))]+\mathbb E[\max(0,1+D(\hat V))]
\]
\[
L_G=-\mathbb E[D(\hat V)]
\]

R1 正则（仅对真实样本）：

\[
L_{R1}=\frac{\gamma}{2}\mathbb E\|\nabla_V D(V)\|_2^2
\]

最终判别器损失：\(L_D^{tot}=L_D+L_{R1}\)。

### 3.1.4 视频总损失

\[
L_{video}=\lambda_{vp}L_{v,pix}+\lambda_{vs}L_{v,st}+\lambda_{vg}L_G+\lambda_{vt}L_{temp}
\]

其中 \(L_{temp}\) 可取光流重投影一致性或特征时序一致。

---

## 3.2 深度分支损失

### 3.2.1 主项：SILog（尺度不变对数损失）

令 \(d_i=\log(\hat d_i)-\log(d_i)\)：

\[
L_{SILog}=\alpha\sqrt{\frac{1}{N_d}\sum_i d_i^2-\frac{\lambda}{N_d^2}\left(\sum_i d_i\right)^2}
\]

- \(N_d\)：有效深度像素数；
- \(\alpha\)：缩放系数；
- \(\lambda\)：去均值项系数（通常接近1）。

**核心推导解释**：第二项消去全局偏移（log域中的平均尺度偏差），使目标更关注相对深度关系。

### 3.2.2 结构项：深度梯度一致

\[
L_{d,grad}=\frac{1}{N_d}\sum_i\left(|\partial_x\hat D_i-\partial_x D_i|+|\partial_y\hat D_i-\partial_y D_i|\right)
\]

**作用**：保持深度边界与局部结构。

### 3.2.3 几何项：法向一致（VNL/normal cosine）

若通过相机内参将深度恢复局部点云并估计法向 \(n_i,\hat n_i\)：

\[
L_{d,norm}=\frac{1}{N_s}\sum_{i=1}^{N_s}\left(1-\frac{\langle n_i,\hat n_i\rangle}{\|n_i\|\,\|\hat n_i\|}\right)
\]

- \(N_s\)：采样点数；
- 近似实现可用固定内参/FOV。

### 3.2.4 深度总损失

\[
L_{depth}=\lambda_{ds}L_{SILog}+\lambda_{dg}L_{d,grad}+\lambda_{dn}L_{d,norm}
\]

---

## 3.3 全模型联合目标（含码率）

设熵模型给出比特估计 \(R\)：

\[
L_{total}=\lambda_v L_{video}+\lambda_d L_{depth}+\lambda_r R+\lambda_{ib}L_{IB}
\]

- \(R\)：可分解为共享码率+私有码率；
- \(L_{IB}\)：仓库已存在 OMIB-like 项，可继续保留并做权重退火。

---

## 4. 指标体系的最优替代（训练外评估）

## 4.1 视频指标（并行而非替换）

- **RD客观**：PSNR / MS-SSIM；
- **感知**：LPIPS（帧级）+ 时序一致指标（tLPIPS或warp error）；
- **工业主观代理**：VMAF；
- **分布质量（研究侧）**：FVD。

> 说明：FVD不应单独作为压缩系统唯一优劣标准，应与RD指标联合报告。

## 4.2 深度指标（标准套件）

\[
AbsRel=\frac{1}{N_d}\sum_i\frac{|\hat d_i-d_i|}{d_i},\quad
SqRel=\frac{1}{N_d}\sum_i\frac{(\hat d_i-d_i)^2}{d_i}
\]
\[
RMSE=\sqrt{\frac{1}{N_d}\sum_i(\hat d_i-d_i)^2},\quad
RMSE_{log}=\sqrt{\frac{1}{N_d}\sum_i(\log\hat d_i-\log d_i)^2}
\]
\[
\delta_t=\frac{1}{N_d}\sum_i\mathbf 1\left(\max\left(\frac{\hat d_i}{d_i},\frac{d_i}{\hat d_i}\right)<1.25^t\right),\ t\in\{1,2,3\}
\]

---

## 5. 代码级修改方案（对应仓库文件）

## 5.1 `losses.py`

1. 新增 `SILogLoss` 类（替换 `DepthLoss` 主项 `L1`）；
2. 在 `DepthLoss` 里组合：`SILog + gradient + normal`；
3. 新增 `SpatiotemporalPerceptualLoss`：输入 `[B,T,C,H,W]`，调用冻结3D backbone；
4. `VideoLoss` 组合：`Charbonnier + 3D perceptual + temporal consistency`；
5. `AdversarialLoss` 从 LSGAN 改为 Hinge，并新增 `R1Regularizer`；
6. 在 `DepthVideoLoss` 中保留 `rate_loss/omib_like_loss`，但新增配置权重项并支持分阶段开关。

## 5.2 `metrics.py`

1. 保留 PSNR/SSIM（建议补充 MS-SSIM）；
2. 默认关闭或移除 BLEU/ROUGE 分支（主任务无文本）；
3. 新增深度：AbsRel, SqRel, RMSE, RMSE(log), δ1/δ2/δ3；
4. 新增视频：VMAF 接口（外部 ffmpeg/libvmaf 调用）与 FVD 评估入口（离线脚本）。

## 5.3 训练策略（建议）

- Stage A（稳态收敛）：\(\lambda_{vg}=0\)，仅 RD + 感知 + 几何；
- Stage B（高保真细节）：逐步增大 \(\lambda_{vg}\) 与 R1 的 \(\gamma\)；
- Stage C（部署前调优）：以目标信道 SNR 分桶调参，获得率失真-感知 Pareto 前沿。

---

## 6. 参数说明与建议初值

- 视频：\(\lambda_{vp}=1.0, \lambda_{vs}=0.05\sim0.2, \lambda_{vt}=0.05\sim0.2, \lambda_{vg}=0\to0.01\)；
- 深度：\(\lambda_{ds}=1.0, \lambda_{dg}=0.1, \lambda_{dn}=0.05\sim0.2\)；
- GAN：R1 的 \(\gamma\) 常见 1~10（按分辨率与batch调优）；
- 码率：\(\lambda_r\) 需按目标 bpp/bps 约束网格搜索。

---

## 7. 2020+ 高权威可访问论文依据

### 视频压缩 / 感知 / 生成稳定性

1. FVC (CVPR 2021): https://arxiv.org/abs/2105.04128  
2. Deep Contextual Video Compression (NeurIPS 2021): https://arxiv.org/abs/2109.15047  
3. High-Fidelity ML Enhanced Video Compression (CVPR 2022): https://arxiv.org/abs/2202.10665  
4. CANF-VC (ECCV 2022): https://arxiv.org/abs/2207.05315  
5. Neural Video Compression with Diverse Contexts (CVPR 2023): https://arxiv.org/abs/2302.14402  
6. VideoMAE (NeurIPS 2022): https://arxiv.org/abs/2203.12602  
7. MagViT (CVPR 2023): https://arxiv.org/abs/2212.05199  
8. Taming Transformers / VQGAN (CVPR 2021): https://arxiv.org/abs/2012.09841  
9. StyleGAN2 Analysis + R1 (CVPR 2020): https://arxiv.org/abs/1912.04958  
10. StyleGAN-V (CVPR 2022): https://arxiv.org/abs/2112.14683  
11. Align Your Latents (CVPR 2023): https://arxiv.org/abs/2304.08818  
12. VMAF: The Journey Continues (TBC 2022): https://ieeexplore.ieee.org/document/9746142

### 深度估计与几何约束

13. ZoeDepth (CVPR 2023): https://arxiv.org/abs/2302.12288  
14. Depth Anything (CVPR 2024): https://arxiv.org/abs/2401.10891  
15. DPT (ICCV 2021): https://arxiv.org/abs/2103.13413  
16. MiDaS TPAMI version (2022): https://arxiv.org/abs/1907.01341  
17. Virtual Normal Loss (TPAMI 2021): https://ieeexplore.ieee.org/document/9357954  
18. NeW CRFs (CVPR 2022): https://arxiv.org/abs/2203.01502  
19. BinsFormer (CVPR 2022): https://arxiv.org/abs/2204.00987

### JSCC 感知导向

20. Towards Perceptual JSCC for Image Transmission (CVPR 2024): https://arxiv.org/abs/2403.07976

---

## 8. 结论

- 对本仓库“最优、时效、权威”的路径不是单一替换，而是 **RD + 感知 + 几何** 三层联合优化；
- 短期先做指标系统与深度主损失升级；中期再做3D感知；长期再做3D对抗分支；
- 该路径能在工程可行性与学术公认度之间取得最优平衡。

