# 多模态 JSCC 改造后整体框架数学推导报告（视频+深度）

## 1. 问题定义

给定源端多模态观测：视频序列 \(V\in\mathbb R^{T\times H\times W\times 3}\) 与深度图 \(D\in\mathbb R^{H\times W}\)。
系统通过联合编码器映射到信道输入符号 \(x\)，经随机信道 \(\mathcal C\) 后得到 \(y\)，在接收端解码重建 \((\hat V,\hat D)\)。

目标是在给定平均传输资源约束下，最小化三类误差：

1. 失真（Rate-Distortion）；
2. 视觉感知（Perceptual Quality）；
3. 深度几何（Geometric Consistency）。

---

## 2. 概率图与信息论形式

设联合编码器与解码器参数分别为 \(\theta_e,\theta_d\)。

\[
z_s,z_v,z_d = E_{\theta_e}(V,D),\quad x = g(z_s,z_v,z_d)
\]
\[
y\sim p(y|x;\text{SNR}),\quad (\hat V,\hat D)=G_{\theta_d}(y)
\]

其中：
- \(z_s\)：跨模态共享潜变量；
- \(z_v,z_d\)：视频/深度私有潜变量；
- \(g\)：调制与功率归一化映射（可并入编码器）。

### 2.1 信息瓶颈视角

最小化任务失真的同时压缩冗余信息，可写成（近似）：

\[
\min_{\theta_e,\theta_d}\; \mathbb E\big[L_{task}(V,D,\hat V,\hat D)\big] + \beta I((V,D);Z)
\]

若采用变分近似（仓库已有 OMIB-like 思路），可替换为 KL 正则之和：

\[
L_{IB}\approx KL(q(z_s|V,D)\|p(z_s)) + KL(q(z_v|V)\|p(z_v)) + KL(q(z_d|D)\|p(z_d))
\]

---

## 3. 改造后联合目标函数推导

## 3.1 任务损失分解

\[
L_{task}=\lambda_v L_{video}+\lambda_d L_{depth}
\]

### 视频项
\[
L_{video}=\lambda_{vp}L_{v,pix}+\lambda_{vs}L_{v,st}+\lambda_{vg}L_G+\lambda_{vt}L_{temp}
\]

### 深度项
\[
L_{depth}=\lambda_{ds}L_{SILog}+\lambda_{dg}L_{d,grad}+\lambda_{dn}L_{d,norm}
\]

## 3.2 率项（可与熵模型对接）

记估计码率为 \(R\)（可用 bpp/bpe 近似）：

\[
R = R_s + \eta_d R_d + \eta_v R_v
\]

- \(R_s\)：共享码流；
- \(R_d,R_v\)：私有码流；
- \(\eta_d,\eta_v\)：动态权重（可由任务损失比自适应得到）。

## 3.3 全局训练目标

\[
L_{total} = \lambda_v L_{video} + \lambda_d L_{depth} + \lambda_r R + \lambda_{ib}L_{IB}
\]

若启用GAN判别器 \(D_\psi\)：

\[
\min_{\theta_e,\theta_d}\max_{\psi}\; L_{total} + \lambda_{adv}\big(\mathbb E[D_\psi(V)]-\mathbb E[D_\psi(\hat V)]\big)-\lambda_{r1}L_{R1}
\]

工程实现上通常拆成 G-step 与 D-step 交替优化。

---

## 4. 关键模块数学解释（参数含义）

## 4.1 视频像素项 \(L_{v,pix}\)

\[
L_{v,pix}=\frac{1}{N_v}\sum_i\sqrt{(v_i-\hat v_i)^2+\epsilon^2}
\]

- 控制参数：\(\epsilon\)；
- 物理意义：稳健拟合信道扰动下的像素误差，保证 RD 基线。

## 4.2 视频时空感知项 \(L_{v,st}\)

\[
L_{v,st}=\sum_{l\in\mathcal L}w_l\|\phi^l_{3D}(V)-\phi^l_{3D}(\hat V)\|_2^2
\]

- 控制参数：层权 \(w_l\)；
- 物理意义：对齐时空语义特征分布，抑制闪烁和运动伪影。

## 4.3 SILog 项 \(L_{SILog}\)

\[
L_{SILog}=\alpha\sqrt{\frac{1}{N_d}\sum_i d_i^2-\frac{\lambda}{N_d^2}(\sum_i d_i)^2},\quad d_i=\log\hat d_i-\log d_i
\]

- \(\alpha\)：尺度系数；
- \(\lambda\)：去均值强度；
- 作用：降低全局尺度偏差影响，强化相对深度一致性。

## 4.4 几何法向项 \(L_{d,norm}\)

\[
L_{d,norm}=\frac{1}{N_s}\sum_{i=1}^{N_s}\left(1-\frac{\langle n_i,\hat n_i\rangle}{\|n_i\|\|\hat n_i\|}\right)
\]

- 需要相机内参 \(K\) 或固定近似内参；
- 作用：约束局部曲面方向，提升3D重建可用性。

## 4.5 率项 \(R\)

若潜变量概率为 \(p(\tilde z)\)，常用近似：

\[
R\approx -\frac{1}{N}\sum_j\log_2 p(\tilde z_j)
\]

- 作用：直接约束传输成本，形成 RD 曲线。

---

## 5. 评估框架的数学定义（改造后）

## 5.1 视频

- RD：PSNR / MS-SSIM；
- 感知：LPIPS；
- 时序与主观代理：VMAF；
- 分布一致：FVD。

FVD 常用形式：
\[
FVD=\|\mu_r-\mu_g\|_2^2+Tr(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})
\]

## 5.2 深度

\[
AbsRel=\frac{1}{N_d}\sum_i\frac{|\hat d_i-d_i|}{d_i},
\quad
\delta_t=\frac{1}{N_d}\sum_i\mathbf 1\Big(\max(\frac{\hat d_i}{d_i},\frac{d_i}{\hat d_i})<1.25^t\Big)
\]

并联合 SqRel / RMSE / RMSE(log) 报告。

---

## 6. 训练与优化流程推导（建议）

1. **预热阶段（无GAN）**：
   \[
   \min\; \lambda_v(\lambda_{vp}L_{v,pix}+\lambda_{vs}L_{v,st})+\lambda_dL_{depth}+\lambda_rR+\lambda_{ib}L_{IB}
   \]
2. **细节增强阶段（启GAN）**：交替更新
   - D-step: 最小化 \(L_D+L_{R1}\)
   - G-step: 最小化 \(L_{total}\) 中含 \(\lambda_{vg}L_G\)
3. **部署调优阶段**：对不同SNR区间做权重重标定，得到 Pareto 前沿。

---

## 7. 变量总表

- \(V,D\)：原始视频/深度；
- \(\hat V,\hat D\)：重建结果；
- \(z_s,z_v,z_d\)：共享/私有潜变量；
- \(R_s,R_v,R_d\)：共享/私有码率；
- \(\lambda_*\)：多目标权重；
- \(\gamma\)：R1 正则系数；
- \(K\)：相机内参矩阵；
- \(\phi_{3D}\)：冻结时空特征网络。

---

## 8. 权威与时效论文依据（可访问）

1. Deep Contextual Video Compression (NeurIPS 2021): https://arxiv.org/abs/2109.15047  
2. FVC (CVPR 2021): https://arxiv.org/abs/2105.04128  
3. High-Fidelity ML Enhanced Video Compression (CVPR 2022): https://arxiv.org/abs/2202.10665  
4. CANF-VC (ECCV 2022): https://arxiv.org/abs/2207.05315  
5. Neural Video Compression with Diverse Contexts (CVPR 2023): https://arxiv.org/abs/2302.14402  
6. VideoMAE (NeurIPS 2022): https://arxiv.org/abs/2203.12602  
7. MagViT (CVPR 2023): https://arxiv.org/abs/2212.05199  
8. StyleGAN2 Analysis + R1 (CVPR 2020): https://arxiv.org/abs/1912.04958  
9. VQGAN (CVPR 2021): https://arxiv.org/abs/2012.09841  
10. VMAF: The Journey Continues (TBC 2022): https://ieeexplore.ieee.org/document/9746142  
11. ZoeDepth (CVPR 2023): https://arxiv.org/abs/2302.12288  
12. Depth Anything (CVPR 2024): https://arxiv.org/abs/2401.10891  
13. DPT (ICCV 2021): https://arxiv.org/abs/2103.13413  
14. Virtual Normal Loss (TPAMI 2021): https://ieeexplore.ieee.org/document/9357954  
15. NeW CRFs (CVPR 2022): https://arxiv.org/abs/2203.01502  
16. BinsFormer (CVPR 2022): https://arxiv.org/abs/2204.00987  
17. Towards Perceptual JSCC for Image Transmission (CVPR 2024): https://arxiv.org/abs/2403.07976

---

## 9. 结论

改造后的统一优化目标将“传输效率（率）—视觉感知（视频）—几何一致（深度）”整合到同一可训练框架，能够在不重写主干架构的前提下，获得更符合 2022+ 学术共识与工程可部署性的结果。

