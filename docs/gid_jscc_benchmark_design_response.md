# GID 数据集接入与统一 JSCC 基准框架设计建议（基于已核验资料）

## 0. 已核验资料与边界

1. 已成功下载并解析 ICCV 2025 论文 **Instance-Level Video Depth in Groups Beyond Occlusions**（后文简称 GID/InstanceDepth 论文）。
2. IEEE 文档 `10843147` 网页直连返回 `Request Rejected`，本地环境无法直接抓取正文。
3. 已克隆并读取 `Uncertainty-guided-UnSSR` 项目 README，确认其“统一范式 + 可插拔 SOTA 模型”的工程思想与“统一 JSCC 基准框架”目标高度一致。

> 因 IEEE 页面受限，以下关于 `10843147` 的引用仅限其 GitHub 仓库 README 中明确可见内容，不扩展到未验证的论文正文细节。

---

## 1) 数据集处理程序应如何重构？

## 1.1 结论（先给答案）

应采用 **“公共数据层 + 方法适配层（Adapter）”** 的两层设计，而不是“每个单模态方法一套独立脚本”。

- 公共层（必须统一）：
  - 样本索引/切分（train/val/test）
  - 信道条件采样（SNR、带宽比、码率预算）
  - 基础解码与时间对齐
  - 统一评测输入输出格式（tensor schema）
- 适配层（方法特异）：
  - 特定模型所需的预处理（归一化、patch/token 化、关键帧策略）
  - 特定标签构造（如实例 mask、深度层、时序先验）
  - 方法专属 collate 与 batch 后处理

这能同时满足“公平比较”和“即插即用复现”。

## 1.2 数学依据（为什么不能每个方法各写一套）

统一基准的目标本质是控制变量实验：

\[
\hat{D}_m = \mathbb{E}_{(x,c)\sim \mathcal{P}}\left[d\big(x,\hat{x}_m(x,c)\big)\right]
\]

其中：
- \(m\) 是方法；
- \(x\) 为同一数据分布样本；
- \(c\) 为同一信道分布；
- \(d\) 为同一失真度量。

若每个方法使用独立脚本，等价于让 \(\mathcal{P}\)（甚至采样细节）随 \(m\) 改变，会引入额外偏置项：

\[
\Delta_{bias} = \big(\mathbb{E}_{\mathcal{P}_m} - \mathbb{E}_{\mathcal{P}_{ref}}\big)[d]
\]

使“方法差异”与“数据管线差异”混杂，不再可公平归因。

因此，应把所有会影响 \(\mathcal{P}\) 与 \(c\) 的因素上收为公共层，仅把模型必要差异下放到 Adapter。

## 1.3 论文依据（与 GID 和 UnSSR 对齐）

- GID 论文强调其数据集是用于**动态多目标、遮挡场景**的严格比较，且给出统一训练/测试设置与统一深度评估指标（RMS、REL、阈值精度等）。这天然要求“统一数据入口 + 统一评测协议”。
- GID 论文的方法结构本身也是“全局初始化 + 实例修正”的分层思想，说明“公共主干 + 专项模块”在工程上可行。
- UnSSR README 明确提出“在统一范式下挂接多种 SOTA 重建网络”的思路，这正对应你提出的“单模态方法插件化集成”。

## 1.4 面向你仓库的落地建议

你当前仓库已有 `MultimodalDataset`，并且已经在同一入口中组织 depth+video、采样策略、严格模式等能力，适合作为“公共层”基础；建议继续保留并抽象出标准接口，而不是推倒重写成“每方法单脚本”。

---

## 2) 损失函数与考核指标如何设计？

## 2.1 结论（先给答案）

你的判断是对的：

- **训练损失**：采用“共同损失 + 方法专属损失”的加权形式；
- **评测指标**：不是“最大公约数（只保留最小交集）”，而是
  1) 一组跨方法可比的 Core 指标（公平对比），
  2) 每方法保留其 Specialty 指标（能力诊断）。

即“双层指标协议”，而非单一最小集合。

## 2.2 数学形式（推荐）

对方法 \(m\) 的训练目标：

\[
\mathcal{L}_m = \lambda_{core}\,\mathcal{L}_{core}
+ \sum_k \lambda_{m,k}\,\mathcal{L}^{(k)}_{spec,m}
+ \beta\,\mathcal{R}_{rate}
\]

其中：
- \(\mathcal{L}_{core}\)：所有方法共享（重建失真 + 速率/带宽约束）；
- \(\mathcal{L}^{(k)}_{spec,m}\)：该方法独有（如时序一致性、实例几何一致性、感知对抗项）；
- \(\mathcal{R}_{rate}\)：传输代价项（带宽、码率、信道使用代价）。

评测分两层：
- Core：所有方法都报告（如 PSNR/SSIM、深度 REL/RMS、视频时序稳定性基础指标）；
- Specialty：按方法额外报告（如 LPIPS、FVD、深度边缘一致性、实例遮挡区域误差）。

## 2.3 与论文的对应关系

- GID/InstanceDepth 在实例修正阶段使用了双损失（`Lobj + Ldist`）并报告其消融，说明“共享主目标 + 专项几何损失”的组合是有效范式，而不是只用单一统一损失。
- GID 在评测上使用多指标（误差类 + 阈值精度类），并未压缩为单一指标，这支持“Core 多指标协议”而非“只留最大公约数”。
- 你当前仓库损失实现已经呈现“重建 + 感知 + 几何/时序”的复合结构，可直接扩展为上述统一形式。

---

## 3) 基准框架的最小可行结构（不引入未验证假设）

以下结构仅依赖你已提出的约束：**融合与信道公用**、单模态方法插件化、在同一框架中训练/评测。

```text
benchmark/
  registry/
    method_registry.py          # 注册单模态JSCC方法
  data/
    core_dataset.py             # 公共读取/切分/采样/对齐
    adapters/
      deepjscc_image.py         # 图像方法适配器
      deepwive_video.py         # 视频方法适配器
      gid_depth_video.py        # GID深度/视频适配器
  channels/
    channel_factory.py          # 公共信道（AWGN/Rayleigh/...）
  models/
    plugins/
      <method_name>/            # 各方法原始或复现实现
    interfaces.py               # encode/transmit/decode标准接口
  training/
    objectives.py               # core + spec loss 组装
    engine.py                   # 统一训练循环
  evaluation/
    core_metrics.py             # 公共指标
    specialty_metrics.py        # 方法专项指标
    protocol.py                 # 统一评测协议
```

统一接口建议：
- `encode(batch, cond) -> latent`
- `channel(latent, snr, bw) -> noisy_latent`
- `decode(noisy_latent, aux) -> recon`
- `compute_losses(outputs, batch) -> dict`
- `compute_metrics(outputs, batch) -> dict`

这样可保证：
1. 信道仿真、融合策略、数据分割在全方法间共享；
2. 单模态模型只需实现接口即可挂载；
3. 能直接做“多模态联合策略 vs 单模态机械组合”的公平实验。

---

## 4. 可直接执行的决策建议

1. **不要**为每个单模态方法重写独立数据脚本。
2. 先抽象“公共 Dataset Core + Adapter API”，再迁移现有加载逻辑。
3. 训练目标采用“Core + Spec + Rate”三部分，权重通过验证集自动寻优（如网格/贝叶斯）。
4. 指标采用“Core 必报 + Specialty 选报 + 统一汇总表”。
5. 报告中必须区分：
   - 同带宽同 SNR 的重建质量优势；
   - 达到同质量时的带宽节省（反向比较）。

---

## 5. 已核验参考来源（真实链接）

- ICCV 2025: *Instance-Level Video Depth in Groups Beyond Occlusions*  
  https://openaccess.thecvf.com/content/ICCV2025/papers/Liang_Instance-Level_Video_Depth_in_Groups_Beyond_Occlusions_ICCV_2025_paper.pdf
- IEEE 文档页（本环境访问受限）：  
  https://ieeexplore.ieee.org/document/10843147
- UnSSR GitHub：  
  https://github.com/SuperiorLeo/Uncertainty-guided-UnSSR.git

