# 模块化修订报告（2026-03-03）

## 1) 审查报告有效性结论

结论：**原审查报告有程序与数学上的可核验依据，且指出的“视频编解码模块/信道模块未完全纳入 modules 体系”问题成立**。

- 程序证据（修订前）：`modules/system.py` 直接依赖根目录实现（`channel.py`, `video_encoder.py`, `video_unet.py`），属于硬编码耦合点。
- 数学证据：
  - MINE 使用 Donsker-Varadhan 下界：
    \[
    I(X;Y) \ge \mathbb E_{P_{XY}}[T] - \log \mathbb E_{P_XP_Y}[e^T]
    \]
  - 率项通过高斯 NLL 近似（单位 bit）：
    \[
    R \approx \mathbb E[-\log_2 p(z)]
    \]
  - 信道层可抽象为：
    \[
    y = h\cdot x + n
    \]

## 2) 本次代码修订内容

### 2.1 新增视频编解码抽象层

新增 `modules/video_codec.py`：
- `BaseVideoEncoder` / `BaseVideoDecoder` 抽象接口；
- `DefaultVideoEncoder` / `DefaultVideoDecoder` 作为对现有实现的薄封装，保持行为兼容。

### 2.2 新增信道抽象层

新增 `modules/channel_models.py`：
- `BaseChannel` 抽象接口；
- `DefaultChannel` 封装现有 `Channel` 实现，保留 `set_snr` 与前向传输能力。

### 2.3 主系统依赖注入改造

`modules/system.py` 的 `DepthVideoJSCC` 新增可注入参数：
- `video_encoder: Optional[BaseVideoEncoder]`
- `video_decoder: Optional[BaseVideoDecoder]`
- `channel: Optional[BaseChannel]`

默认行为仍由 `DefaultVideoEncoder` / `DefaultVideoDecoder` / `DefaultChannel` 提供，因此训练与推理流程保持向后兼容。

### 2.4 统一导出层更新

- `modules/__init__.py` 与 `multimodal_jscc.py` 增加新抽象类与默认实现导出，便于外部替换组件而不改主系统。

## 3) 数学与论文支撑（真实链接）

1. **MINE（互信息估计）**
   - Belghazi et al., ICML 2018.
   - 链接：https://arxiv.org/abs/1801.04062

2. **变分压缩/超先验率失真建模**
   - Ballé et al., ICLR 2018.
   - 链接：https://arxiv.org/abs/1802.01436

3. **视频压缩的模块化分解（运动/补偿/残差）**
   - Lu et al., DVC, CVPR 2019.
   - 链接：https://arxiv.org/abs/1812.00101

4. **可微信道层（AWGN/Rayleigh）在 JSCC 中的角色**
   - Bourtsoulatze et al., TCCN 2019.
   - 链接：https://arxiv.org/abs/1809.01733

## 4) 影响评估

- **功能等价性**：默认实例化路径与原实现一致，输出接口与关键统计项不变。
- **可扩展性提升**：后续可在不改 `DepthVideoJSCC` 主类的情况下，替换为其它视频编解码器或信道模型（如状态空间时序建模、带记忆信道等）。
- **风险控制**：依赖注入采用“可选参数 + 默认实现”策略，避免破坏已有训练脚本。
