# OMIB-like 可落地改造实施报告

## 参考链接（真实）
- OMIB 论文摘要：https://arxiv.org/abs/2505.19996
- OMIB 论文 PDF（Eq.(10)/(11)/(15)/(17), Prop.5.1/5.2/5.7）：https://arxiv.org/pdf/2505.19996
- MINE（Mutual Information Neural Estimation）：https://arxiv.org/abs/1801.04062
- Deep Variational Information Bottleneck（变分 IB）：https://arxiv.org/abs/1612.00410

---

## 1) 所做修改位置及内容

### 1. `multimodal_jscc.py`（`DepthVideoJSCC`）
- 新增参数：
  - `enable_omib_stats: bool = True`
  - `omib_eps: float = 1e-6`
- 在 `DepthVideoJSCC.forward` 末尾新增 `omib_stats` 输出：
  - 从 `depth_private` 计算经验高斯参数 `depth_mu, depth_logvar`
  - 从 `video_private` 计算经验高斯参数 `video_mu, video_logvar`
  - 计算闭式 KL：
    \[
    \mathrm{KL}(\mathcal N(\mu,\sigma^2)\|\mathcal N(0,1))
    =\frac{1}{2}(\mu^2+\sigma^2-1-\log\sigma^2)
    \]
  - 输出 `depth_kl`、`video_kl`（均值标量）供损失函数使用。

### 2. `losses.py`（`DepthVideoLoss`）
- 扩展构造参数：
  - `use_omib_like`, `ib_beta`, `ib_beta_min`, `ib_beta_max`, `omib_eps`
- 新增 OMIB-like 组件：
  1. `r` 动态权重（与论文 Eq.(11) 同类的“比值 + tanh/log”结构）
     - 用当前可观测任务损失（depth/video 重建项）构造比值并计算 `r`：
       \[
       r=1-\tanh\left(\log\frac{L_{video}}{L_{depth}}\right),\ r\in[0,2]
       \]
  2. 模态自适应 rate 权重：
     - 从 `r` 得到 `\lambda_d, \lambda_v`，将原统一 rate 惩罚改为分模态惩罚：
       \[
       R=R_s+\lambda_d R_d+\lambda_v R_v
       \]
  3. OMIB-like KL 正则：
       \[
       L_{omib-like}=\beta\left(\mathrm{KL}_d+r\,\mathrm{KL}_v\right)
       \]
     - 其中 `\beta` 应用上下界裁剪（`ib_beta_min` / `ib_beta_max`）。
- 日志项新增：`omib_dynamic_r`, `lambda_depth`, `lambda_video`, `omib_like_loss`, `omib_beta_eff` 等。

### 3. `train.py`
- `create_model` 传入 `enable_omib_stats`。
- `create_loss_fn` 传入 `use_omib_like`, `ib_beta`, `ib_beta_min`, `ib_beta_max`。
- 新增 CLI 参数：
  - `--use-omib-like`
  - `--ib-beta`
  - `--ib-beta-min`
  - `--ib-beta-max`
- 启动时支持将以上参数写入 `config`。

### 4. `config.py`
- 新增默认配置项：
  - `use_omib_like = True`
  - `ib_beta = 1e-4`
  - `ib_beta_min = 0.0`
  - `ib_beta_max = None`

---

## 2) 所作修改的数学与论文依据

1. **OMIB 目标（两模态）**
   \[
   \min_{\xi}\ -I(\xi;y)+\beta\left(I(\xi;v_1)+rI(\xi;v_2)\right)
   \]
   对应论文 Eq.(15)/(17)。

2. **可训练的变分形式**
   OMIB 使用任务项与 KL 正则并联（论文 Eq.(10), Prop.5.1）。
   本次改造在仓库中对应为：
   - 任务项：现有 depth/video 重建损失
   - KL 项：新增 `depth_kl`、`video_kl` 与 `r` 加权

3. **动态模态权重 `r`**
   论文给出显式可计算 `r`（Eq.(11), Prop.5.2），核心是“模态相对信息缺口驱动正则强度”。
   本实现采用同类函数族 `1-tanh(log(比值))` 作为可落地近似，确保可计算与可稳定训练。

4. **`\beta` 约束思想**
   论文在理论分析中给出 `\beta` 可行区间与上界思想（Sec.5.2, Prop.5.7）。
   本实现提供 `ib_beta_min/max` 机制以支持工程侧区间控制；
   若后续引入 MINE 估计 `H(v_i), I(v_1;v_2)`，可据数据估计进一步设置上界。

---

## 3) 所做修改的作用

1. **将统一 rate 惩罚改成模态自适应 rate 惩罚**
   - 原先 `rate_weight * mean(rate_stats)` 无法体现模态不平衡。
   - 新实现通过 `r` 派生 `\lambda_d, \lambda_v`，在运动剧烈/深度变化显著等情形可自适应调节惩罚分配。

2. **补充 OMIB-like 的可训练 KL 正则**
   - 原先 `DepthVideoLoss` 仅做重建 + rate。
   - 新实现增加 `\beta(\mathrm{KL}_d+r\mathrm{KL}_v)`，把“压缩性约束”从纯 rate 代理扩展到潜变量分布约束。

3. **把改造接入训练主流程并可配置化**
   - 默认可直接启用（配置项 + CLI）
   - 可通过 `--no-use-omib-like` 或调整 `ib_beta` 区间做消融实验。

> 说明：本次实现是“OMIB-like 可落地改造”，在未完整复现论文全部训练结构（例如论文中的任务头和其 KL 比值定义细节）之前，不宣称与论文理论命题等价。

