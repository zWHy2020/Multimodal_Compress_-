# OMIB 优化二次修订完整报告（基于当前仓库代码）

## 0. 真实参考链接

- OMIB 论文（ICML 2025）：https://arxiv.org/abs/2505.19996
- OMIB PDF（含 Eq.(10)/(11)/(15)/(17), Prop.5.1/5.2/5.7）：https://arxiv.org/pdf/2505.19996
- MINE（Mutual Information Neural Estimation）：https://arxiv.org/abs/1801.04062
- Deep Variational Information Bottleneck（变分 IB）：https://arxiv.org/abs/1612.00410
- 互信息定义与恒等式（信息论基础）：https://en.wikipedia.org/wiki/Mutual_information

---

## 1) 对你给出的“现状报告”逐条核验（数学依据 + 程序依据 + 论文依据）

### 1.1 “特征独立性假设未完成”——原结论在旧版本中成立

- 旧代码中 `DepthVideoLoss` 的 OMIB-like 项为
  \[
  L_{omib-like}=\beta(\mathrm{KL}_d+r\,\mathrm{KL}_v)
  \]
  且 rate 聚合主要是可分项求和，确实没有显式互信息修正项。
- 这与 OMIB 在信息量表达中使用的
  \[
  H(v_1)+H(v_2)-I(v_1;v_2)
  \]
  还不一致（OMIB Sec.5.2）。

### 1.2 “模态不平衡权重已完成（近似）”——成立

- 代码通过
  \[
  r=1-\tanh\left(\log\frac{L_{video}}{L_{depth}}\right),\ r\in[0,2]
  \]
  构造动态权重，并用于 rate 分项与 KL 分项加权。
- 这与 OMIB 的核心思想（Eq.(11), Prop.5.2：动态调节模态相对正则化）一致，属于可落地近似。

### 1.3 “理论上界保证未完成（仅工程裁剪）”——原结论在旧版本中成立

- 若无 `H(v_i)`/`I(v_1;v_2)` 的数据估计，就无法按 OMIB 的理论区间思路计算上界（Sec.5.2, Prop.5.7）。
- 仅 `ib_beta_min/max` 的固定裁剪，确属工程策略，不是数据驱动上界推导。

---

## 2) 本次代码更改（针对 1.1 与 1.3 的补全）

## 2.1 引入跨模态互信息估计并用于联合码率修正（补全 1.1）

### 修改位置
- `multimodal_jscc.py`
  - 新增 `MineEstimator`（DV 形式 MINE 下界估计）
  - `DepthVideoJSCC` 新增参数：`enable_mi_correction`, `mine_hidden_dim`
  - 在 `forward` 中计算 `cross_modal_mi_bits` 并写入 `rate_stats` / `entropy_stats`
- `losses.py`
  - `DepthVideoLoss` 新增 `mi_correction_weight`
  - 在 rate 聚合中从
    \[
    R_s+\lambda_dR_d+\lambda_vR_v
    \]
    改为
    \[
    R_s+\lambda_dR_d+\lambda_vR_v-\alpha I_{12}
    \]
    其中 `I_{12}` 来自 MINE 估计（`cross_modal_mi_bits`），`\alpha=mi_correction_weight`。

### 数学依据
- 互信息恒等式：联合信息可由边缘信息减去互信息重叠项（信息论基础）。
- OMIB 的边界项中直接使用 `H(v_1)+H(v_2)-I(v_1;v_2)`，本次把 `-I` 作为率项修正显式落地。

### 作用
- 不再把两模态私有信息完全当作独立可加，显式引入“重叠信息扣减”。

## 2.2 增加基于 MINE 的 beta 上界估计流程（补全 1.3）

### 修改位置
- 新增 `mine_utils.py`
  - `estimate_beta_upper_bound_mine(...)`
  - 用三组 MINE 估计：
    - `I(v_1;v_2)`
    - `H(v_1)=I(v_1;v_1)`
    - `H(v_2)=I(v_2;v_2)`
  - 计算
    \[
    M_u=\frac{1}{3\left(H(v_1)+H(v_2)-I(v_1;v_2)\right)}
    \]
- `train.py`
  - 启动后、正式训练前，若 `use_mine_beta_bound=True`，调用上述估计函数；
  - 将 `config.ib_beta_max` 更新为估计上界（与已有手动上界取更严格者）。
- `config.py`
  - 新增 `use_mine_beta_bound`, `mine_beta_estimate_steps`, `mine_train_steps`, `mine_hidden_dim`, `mine_lr`
- `train.py` CLI
  - 新增 `--use-mine-beta-bound`, `--mine-beta-estimate-steps`, `--mine-train-steps`, `--mine-hidden-dim`, `--mine-lr`, `--mi-correction-weight`

### 数学依据
- OMIB Sec.5.2 / Prop.5.7 给出 `beta` 上界思想；
- OMIB Appendix E 使用 MINE 估计熵与互信息（`H(X)=I(X;X)`）。
- 本实现按该思路估计并映射到 `ib_beta_max`，避免纯手动设置。

### 作用
- `ib_beta_max` 从“仅人工裁剪”升级为“可数据驱动估计 + 人工上限并存”的机制。

---

## 3) 文件级修改清单

1. `multimodal_jscc.py`
- 新增 `MineEstimator`
- `DepthVideoJSCC` 新增 MI 修正开关与 `cross_modal_mi_bits` 输出

2. `losses.py`
- `DepthVideoLoss` 新增 `mi_correction_weight`
- rate 项加入 `- alpha * cross_modal_mi_bits`

3. `mine_utils.py`（新文件）
- 新增 MINE 估计与 `M_u` 计算工具

4. `config.py`
- 新增 MINE 与 MI 修正相关配置默认项

5. `train.py`
- 训练前加入 MINE 上界估计流程
- 新增 CLI 与 config 映射

---

## 4) 与“不要编造/不要假设”的一致性说明

- 本次实现没有声称“完全复现 OMIB 全部理论条件”；仅把论文中**可直接工程实现**的两条关键机制（互信息修正、MINE 上界估计）显式接入当前代码。
- 所有公式与方法来源均给出真实链接，并在代码中有对应实现路径。

