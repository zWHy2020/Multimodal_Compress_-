# 多模态 JSCC（视频+深度）损失与指标审查复核（2026）

## 1. 针对现有仓库实现的程序级复核结论

### 1.1 现有实现客观事实（来自代码）

- 视频重建主损失目前是逐像素 `L1`，并包含“帧差 L1”时序项。`VideoLoss.forward` 里核心是 `F.l1_loss(pred_f32, target_f32)` 与 `F.l1_loss(pred_diff, target_diff)`。  
- 感知损失 `PerceptualLoss` 仅使用 2D LPIPS(AlexNet) 或 2D VGG 特征，不包含时空 3D 特征。  
- 对抗损失 `AdversarialLoss` 是 LSGAN（MSE 到 real/fake 标签），不是 Hinge 或 NS-GAN+R1。  
- 深度损失 `DepthLoss` 是 `L1 + 2D 梯度一致性`，没有尺度不变项、没有法向几何约束。  
- 指标模块实现了 PSNR/SSIM，同时仍保留 BLEU/ROUGE 文本指标并在 `calculate_multimodal_metrics` 中可被触发。  

> 以上事实与审查报告主判断一致：当前实现偏 2016-2019 的“像素失真主导”范式，对视频时序感知与深度几何建模确实不足。

### 1.2 对审查报告“是否合理”的逐条判断

**(A) “VideoLoss 时效性差”——总体合理，但需补充边界条件**  
- 合理点：在高压缩/低码率 JSCC 下，纯 L1/L2 常造成过平滑，这是神经压缩文献共识。  
- 边界点：在率失真（RD）优化里，L1/L2 仍是主流 distortion 基线，并非“不可用”；它通常需要与感知项/对抗项/特征项联合。

**(B) “2D 感知损失用于视频存在模态错配”——合理**  
- 2D LPIPS/VGG 逐帧约束无法直接约束时序一致性，容易产生 temporal flicker。使用 I3D/VideoMAE 等时空 backbone 更符合视频分布建模。

**(C) “LSGAN 脆弱，应升级”——基本合理**  
- 2020 后视觉生成侧主流确实偏向 Hinge / Non-saturating + R1（或 GP）稳定化。对于视频，3D 判别器或多帧判别器比 2D 判别器更匹配。

**(D) “DepthLoss 缺乏尺度不变与3D几何”——合理**  
- 仅 L1 + 图像梯度不显式建模深度的尺度不变性与法向几何，难以兼顾近距/远距误差。

**(E) “应直接移除 PSNR/SSIM”——该结论过强，建议修正**  
- 现代感知范式确实强调 LPIPS/FVD/VMAF 等；但神经压缩社区在 RD 曲线、BD-rate 对比中仍大量报告 PSNR/MS-SSIM。  
- 更合理做法：**不要只用 PSNR/SSIM**，但也不建议“完全移除”。应保留作客观失真参照，并新增感知与任务相关指标。

**(F) “BLEU/ROUGE 在该仓库场景无效”——合理**  
- 该仓库目标为视频+深度联合压缩传输，当前任务描述不包含文本解码评价；BLEU/ROUGE 对主目标无贡献，易造成指标噪声。

---

## 2. 数学与理论依据（面向仓库目标）

### 2.1 为什么要从“像素误差”升级到“时空语义+几何”

1. **感知-失真折中（Perception-Distortion Tradeoff）**：仅优化像素误差会偏向均值解，导致纹理细节损失。  
2. **视频的时空耦合性**：帧内误差最小并不保证帧间稳定；需要显式时序项或3D特征项。  
3. **深度的相对尺度属性**：绝对误差并不等价于几何误差；深度任务常用对数域与比例域指标。  

### 2.2 推荐的损失形式（可直接映射到现有代码）

- **视频重建主项（稳健像素）**：Charbonnier / L1（保留）。  
  \[
  L_{pix}=\frac{1}{N}\sum_i \sqrt{(x_i-\hat{x}_i)^2+\epsilon^2}
  \]
- **视频时空感知项（I3D/VideoMAE 特征）**：  
  \[
  L_{st}=\|\phi_{3D}(V)-\phi_{3D}(\hat V)\|_2^2
  \]
- **视频对抗项（Hinge + R1）**：  
  \[
  L_D=\mathbb{E}[\max(0,1-D(V))]+\mathbb{E}[\max(0,1+D(\hat V))],\quad
  L_{R1}=\frac{\gamma}{2}\mathbb{E}\|\nabla_V D(V)\|_2^2
  \]
- **深度主项（SILog）**：令 \(d_i=\log y_i-\log \hat y_i\)，  
  \[
  L_{SILog}=\alpha\sqrt{\frac{1}{N}\sum_i d_i^2-\frac{\lambda}{N^2}(\sum_i d_i)^2}
  \]
- **深度几何项（法向一致性，可用 VNL 或 surface normal loss）**：  
  \[
  L_{normal}=\frac{1}{N}\sum_i (1-\cos\langle n_i,\hat n_i\rangle)
  \]

---

## 3. 对“替代方案是否最优”的复核（结合 2022+ 主流）

结论：原审查给出的方向**大体正确**，但“最优”应修正为“按任务分层组合最优（RD+感知+几何）”，而非单指标替换。

### 3.1 视频损失：2022+ 更稳妥的公认组合

**推荐组合（训练）**  
\[
L_{video}=\lambda_1L_{pix}+\lambda_2L_{st}+\lambda_3L_{GAN}+\lambda_4L_{temp}
\]
- `L_pix`：L1/Charbonnier（保 RD 基座）  
- `L_st`：I3D 或 VideoMAE 时空特征损失（替代逐帧2D感知）  
- `L_GAN`：Hinge + R1（若启用判别器）  
- `L_temp`：光流/特征时序一致性（可选，预算不足可先不用）

**推荐组合（评估）**  
- RD 向：PSNR / MS-SSIM（保留）  
- 感知向：LPIPS (frame) + tLPIPS/warp error  
- 视频主观代理：VMAF（工程可用）  
- 生成/分布一致：FVD（研究可用）

> 注意：FVD 更偏生成分布质量，不应单独作为压缩重建唯一指标。

### 3.2 深度损失：2022+ 公认组合

**推荐组合（训练）**  
\[
L_{depth}=\beta_1L_{SILog}+\beta_2L_{grad}+\beta_3L_{normal}
\]
- `L_SILog`：主损失（尺度不变）  
- `L_grad`：边缘/局部结构保真（保留当前梯度项思想）  
- `L_normal`：几何约束（VNL 或法向余弦损失）

**推荐组合（评估）**  
- AbsRel, SqRel, RMSE, RMSE(log), \(\delta_1,\delta_2,\delta_3\)（单目深度主流）

### 3.3 对仓库的“时效性最强+公认性最好”落地优先级

1. **立即执行（低风险高收益）**：
   - 去除默认报告中的 BLEU/ROUGE（或至少默认关闭）。
   - 新增深度指标：AbsRel + \(\delta\) 系列。
   - 保留 PSNR/SSIM 但增加 VMAF/FVD（评估脚本层，不入反传）。
2. **第二阶段（中成本）**：
   - 将 `DepthLoss` 升级为 `SILog + gradient`。
   - 将 `PerceptualLoss` 从 2D 切到 3D backbone 特征损失。
3. **第三阶段（中高成本）**：
   - 对抗分支升级为 Hinge+R1，并将判别器改为 3D/多帧 Patch 判别器。

---

## 4. 2020+（尤其 2022+）可访问论文依据（精选，真实链接）

> 下列为与本仓库目标最相关、且在社区广泛使用或高影响的依据；用于支撑上面的替代建议。

1. FVC, CVPR 2021: https://arxiv.org/abs/2105.04128  
2. Deep Contextual Video Compression, NeurIPS 2021: https://arxiv.org/abs/2109.15047  
3. High-Fidelity ML Enhanced Video Compression, CVPR 2022: https://arxiv.org/abs/2202.10665  
4. CANF-VC, ECCV 2022: https://arxiv.org/abs/2207.05315  
5. Neural Video Compression with Diverse Contexts, CVPR 2023: https://arxiv.org/abs/2302.14402  
6. VideoMAE, NeurIPS 2022: https://arxiv.org/abs/2203.12602  
7. MagViT, CVPR 2023: https://arxiv.org/abs/2212.05199  
8. Taming Transformers (VQGAN), CVPR 2021: https://arxiv.org/abs/2012.09841  
9. StyleGAN2 analysis + R1, CVPR 2020: https://arxiv.org/abs/1912.04958  
10. StyleGAN-V, CVPR 2022: https://arxiv.org/abs/2112.14683  
11. Align Your Latents, CVPR 2023: https://arxiv.org/abs/2304.08818  
12. ZoeDepth, CVPR 2023: https://arxiv.org/abs/2302.12288  
13. Depth Anything, CVPR 2024: https://arxiv.org/abs/2401.10891  
14. DPT, ICCV 2021: https://arxiv.org/abs/2103.13413  
15. MiDaS (TPAMI 2022 version): https://arxiv.org/abs/1907.01341  
16. Virtual Normal Loss, TPAMI 2021: https://ieeexplore.ieee.org/document/9357954  
17. NeW CRFs, CVPR 2022: https://arxiv.org/abs/2203.01502  
18. BinsFormer, CVPR 2022: https://arxiv.org/abs/2204.00987  
19. VMAF Journey Continues, TBC 2022: https://ieeexplore.ieee.org/document/9746142  
20. Toward Perceptual JSCC for Image Transmission, CVPR 2024: https://arxiv.org/abs/2403.07976  

---

## 5. 最终回答（简版）

- 审查报告对仓库现状“落后于最新视频/深度联合感知压缩实践”的判断是**基本准确**的。  
- 但“完全抛弃 PSNR/SSIM”不是最优实践；对压缩任务应采用“传统失真 + 感知 + 任务几何”三组指标并行。  
- 对本仓库当前代码，最优升级路径是：
  1) 先清理不相关 NLP 指标并补齐深度/视频现代评估；
  2) 再把 `DepthLoss` 升级到 SILog 系，`PerceptualLoss` 升级到 3D 特征；
  3) 最后再做 3D 判别器 + Hinge/R1（收益大但工程成本更高）。

