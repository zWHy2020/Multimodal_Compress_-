# GID-Driven Unified JSCC Benchmark 框架改造工作日志

## 1. 修改目的

1. 在**不急于集成具体单模态 SOTA 方法**（如 Deep JSCC / DeepWiVe）的前提下，先建立统一基准框架的“骨架层”：
   - 标准化方法接口（encode/decode/specialized loss）；
   - 可插拔方法注册机制；
   - 统一核心训练目标（失真+速率代理）；
   - 面向 GID 的统一数据读取模块。
2. 将你补充的 GID 文件结构（`images/`, `aligned_depths/`, `instance-masks/`, `video-train.txt`, `video-test.txt`）落地到可执行的数据索引与加载代码。
3. 保证后续“多模态联合方法 vs 单模态机械组合”实验具备同一数据分布、同一信道条件、同一评估协议的可复现实验基础。

---

## 2. 实际修改内容

### 2.1 新增统一 JSCC 基准框架骨架

- `benchmark/interfaces.py`
  - 定义 `JSCCMethodProtocol`：`encode`, `decode`, `compute_specialized_loss`。
  - 定义 `JSCCMethodOutput`：统一输出重建结果、速率代理和辅助张量。

- `benchmark/models/registry.py`
  - 新增 `JSCCMethodRegistry`，用于方法工厂注册和实例化。
  - 当前阶段只做框架层，不集成具体单模态方法实现，符合“本次重在搭框架”的要求。

- `benchmark/training/objectives.py`
  - 新增 `build_core_loss`：
    \[
    \mathcal{L}_{core}=\mathrm{MSE}(\hat{x},x)+\lambda_{rate}\hat{R}
    \]
  - 其中 \(\hat{R}\) 为统一速率代理项（可由各方法提供）。

### 2.2 新增 GID 数据集统一读取模块

- `benchmark/data/gid_dataset.py`
  - 新增 `GIDFramePairDataset`：
    - 根据 `video-train.txt` / `video-test.txt` 索引视频序列；
    - 从 `images/<video>/<frame>.jpg` 与 `aligned_depths/<video>/<frame>.png` 做严格配对；
    - 默认仅返回 `rgb` 与 `depth`（符合你当前训练主需求）；
    - `use_instance_masks=False` 为默认，支持未来切换到 `True` 时加载 `instance-masks`。
  - 深度张量按 10m 上限进行尺度映射并 `clamp[0.01, 10.0]`，与论文描述的深度范围保持一致。

---

## 3. 关于“instance-masks 对当前模型意义不大”的判断

## 3.1 结论（分阶段）

- **对你当前这一阶段目标（先搭统一 JSCC 基准骨架）**：你的观点是**成立的**。只用 `images + aligned_depths` 就能完成“统一信道、统一协议、统一输入输出接口”的主流程。
- **对后续要严格证明遮挡场景优势**：instance masks 仍然有重要价值，建议保留为可选分支而不是彻底弃用。

## 3.2 论文依据

1. GID/InstanceDepth 论文明确将实例信息用于第二阶段“Instance-Aware Depth Rectification”，并使用实例关系损失（`Lobj` 与 `Ldist`）处理遮挡歧义；这说明在“遮挡敏感评估/训练”里，实例掩码是有信息增益的。  
   链接：
   - ICCV OpenAccess PDF：
     https://openaccess.thecvf.com/content/ICCV2025/papers/Liang_Instance-Level_Video_Depth_in_Groups_Beyond_Occlusions_ICCV_2025_paper.pdf

2. Mask2Former 将掩码预测作为通用分割范式，说明 instance mask 具备稳定的对象级几何表达能力，适合做遮挡区域约束。  
   链接：
   - arXiv:2112.01527
     https://arxiv.org/abs/2112.01527

## 3.3 数学依据

把你的当前设置写成目标函数：
\[
\min_\theta\;\mathbb{E}[d(\hat{x}_\theta, x)] + \beta R
\]

这里仅需 RGB 与 depth 监督即可训练/比较 JSCC 链路，因此“可运行、可公平比较”并不依赖 mask。

但在遮挡区域，若引入实例先验 \(m\)（mask）可形成条件优化：
\[
\min_\theta\;\mathbb{E}[d(\hat{x}_\theta, x)\mid m] + \beta R + \lambda\,\mathcal{L}_{occ}
\]

当评价目标强调 occlusion robustness 时，\(m\) 能降低“前后景混叠”造成的估计偏差。因此，mask 不应在框架层被删除，而应作为**可插拔监督源**。

---

## 4. 本次修改的论文与数学依据（真实链接）

1. GID 数据集与 InstanceDepth 方法（实例遮挡建模、损失与指标）  
   https://openaccess.thecvf.com/content/ICCV2025/papers/Liang_Instance-Level_Video_Depth_in_Groups_Beyond_Occlusions_ICCV_2025_paper.pdf

2. Mask2Former（实例掩码与统一分割建模）  
   https://arxiv.org/abs/2112.01527

3. Deep JSCC（端到端语义通信代表工作）  
   https://arxiv.org/abs/1809.01733

4. DeepWiVe（视频 JSCC 代表工作）  
   https://arxiv.org/abs/2106.02797

5. 速率-失真理论基础（Shannon）  
   https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf

---

## 5. 产出与后续

- 本次已完成“统一框架骨架 + GID 数据层”改造。
- 下一步（你确认后）可在不改公共层的前提下逐步接入单模态方法插件，并复用统一训练/评估协议做公平对比。
