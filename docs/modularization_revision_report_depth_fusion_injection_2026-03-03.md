# 深度编解码与融合模块注入修订报告（2026-03-03）

## 1. 审查报告正确性判断

结论：**审查报告正确**。在上一版修改中，`DepthVideoJSCC` 已支持视频编解码器与信道注入，但深度编解码与融合子模块仍是具体类硬编码，解耦不对称。

程序依据（可在仓库中直接验证）：
- 系统主类的默认结构中，深度端与融合端是核心可替换组件；若不提供注入接口，会限制试验空间。
- 现有融合公式与码率项、MINE 估计具有明确的数学对象和边界条件，适合通过接口抽象进行模块化复用。

## 2. 本次进一步修改

### 2.1 深度编解码抽象

新增 `modules/depth_models.py`：
- `BaseDepthEncoder` / `BaseDepthDecoder` 抽象接口；
- `DefaultDepthEncoder` / `DefaultDepthDecoder` 封装原 `DepthJSCCEncoder` / `DepthJSCCDecoder`，确保默认行为不变。

### 2.2 融合/熵模型/MI 抽象

新增 `modules/fusion_models.py`：
- `BaseJointFusion`、`BaseEntropyModel`、`BaseMineEstimator` 抽象接口；
- `DefaultJointFusion`、`DefaultEntropyModel`、`DefaultMineEstimator` 封装原实现。

其中 `BaseJointFusion` 增加了
- `project_shared_to_depth(shared, like)`
- `project_shared_to_video(shared, like)`

避免 `DepthVideoJSCC` 直接依赖融合实现的内部层名（如 `shared_to_depth`），降低耦合。

### 2.3 主系统依赖注入扩展

`modules/system.py` 的 `DepthVideoJSCC` 新增可注入项：
- `depth_encoder`, `depth_decoder`
- `joint_fusion`, `entropy_model`, `mine_estimator`

并保持默认回退到 `Default*` 实现，以保证向后兼容。

### 2.4 统一导出层更新

`modules/__init__.py` 与 `multimodal_jscc.py` 已导出新接口与默认实现，便于外部替换。

## 3. 数学与论文依据（真实可访问链接）

1. 互信息下界（MINE, DV 形式）
- Belghazi et al., ICML 2018.
- https://arxiv.org/abs/1801.04062

2. 率失真中的高斯先验 NLL 近似
- Ballé et al., ICLR 2018.
- https://arxiv.org/abs/1802.01436

3. 视频压缩的模块化（运动估计/补偿/残差编码）
- Lu et al., DVC, CVPR 2019.
- https://arxiv.org/abs/1812.00101

4. JSCC 中可微信道层（AWGN/Rayleigh）
- Bourtsoulatze et al., TCCN 2019.
- https://arxiv.org/abs/1809.01733

5. 深度信息的边缘与平滑先验（TV 正则经典文献）
- Rudin, Osher, Fatemi, 1992.
- https://www.researchgate.net/publication/222469583_Nonlinear_total_variation_based_noise_removal_algorithms

6. 图拉普拉斯与图信号处理基础
- Shuman et al., IEEE Signal Processing Magazine, 2013.
- https://arxiv.org/abs/1211.0053

> 注：第 5/6 条用于说明“为什么深度编码器需要可替换接口”这一建模动机，不代表当前代码已实现 TV 或 GNN 编码器。

## 4. 影响评估

- 功能兼容：默认构造路径保持原模型行为；
- 解耦增强：深度端与融合端与视频端、信道端在抽象层面达到对称；
- 研究可扩展性：可以不改 `DepthVideoJSCC` 主类，即插即用替换编码器、融合器、熵模型、MI 估计器。
