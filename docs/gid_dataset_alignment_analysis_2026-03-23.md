# GID 数据集在本仓库中的用途分析（基于已上传样例与 ICCV 2025 InstanceDepth）

## 1. 可直接核验的仓库现状

### 1.1 已存在的论文与报告文件
仓库根目录已包含以下两个 PDF：
- `Liang_Instance-Level_Video_Depth_in_Groups_Beyond_Occlusions_ICCV_2025_paper.pdf`
- `zWHy2020_Multimodal_Compress_- 代码库深研与 ICCV2025 InstanceDepth 论文对齐分析报告.pdf`

### 1.2 已存在的样例标注/样例帧
仓库根目录已包含：
- `badminton_205641.json`
- `00000.jpg`
- `00000.png`

其中 `badminton_205641.json` 的顶层 key 为帧号字符串（例如 `"00000"` 到 `"00301"`），每帧 value 是一个对象列表；对象条目形如
`[x1, y1, x2, y2, ?, id]`（示例：`[358.0, 204.0, 401.0, 353.0, 0.0, 2.0]`）。
这说明该 JSON 提供逐帧实例框 + 实例身份编号信息。

### 1.3 训练/测试划分文件
仓库已包含 `video-train.txt` 和 `video-test.txt`，且列表项包含你补充提到的序列命名（例如 `badminton_205641`、`badminton_205654`、`dance_202238`、`basketball_192554`）。

## 2. 与 ICCV 2025 论文信息的一致性

依据论文正文可提取文本（本地 PDF）可核验到：
1. 论文提出 GID（Group Instance Depth）数据集，规模为 101,500 帧；
2. 任务重点是“多目标+遮挡”场景下实例级视频深度；
3. 方法分两阶段（Holistic Depth Initialization + Instance-Aware Depth Rectification）；
4. 方法中显式使用实例 mask、实例层深度与遮挡关系约束（含成对相对深度误差项）。

这与你补充的数据结构（`aligned_depths/images/instance-masks/box2d` + 每视频逐帧对齐）是同一任务范式：
- RGB 帧提供外观；
- aligned depth 提供监督；
- instance mask 与 box2d 提供实例几何/遮挡结构。

## 3. 该数据集在你当前 GitHub 项目的“实际用途”

### 用途 A：作为多模态 JSCC 的标准化输入源（RGB + Depth）
你的仓库核心目标是联合传输/压缩视频与深度。GID 的逐帧 RGB-Depth 对齐可直接构成模型的双模态输入对，适合做：
- 端到端 RD（rate-distortion）训练；
- 信道噪声下（AWGN/衰落）鲁棒性评测；
- 复杂遮挡场景下误码对深度/视频重建质量影响分析。

### 用途 B：把“实例结构”引入当前损失与评测
GID 额外提供实例 mask 与 box2d，因此可在现有 pipeline 中扩展：
- 按实例区域统计深度误差（instance-wise REL/RMSE/SILog）；
- 统计“遮挡边界附近”误差（occlusion-focused metric）；
- 对码率分配策略进行实例敏感分析（例如前景实例 vs 背景区域）。

通俗解释：
- 现在很多评测是“整张图平均分”，这会把“人脸/球拍/手臂被遮挡区域”的严重错误冲淡。
- 有了 instance-mask，你可以把每个对象单独拎出来算误差，比如“运动员A 的深度误差”和“背景地板误差”分开看。
- 有了 box2d，你还能知道每一帧有哪些对象、位置大概在哪里，便于把“重叠区域”当成重点区域做专项评测。

一个最小落地流程（不改模型结构）：
1. 正常跑你的深度重建，得到预测深度 `\hat d` 与 GT 深度 `d`。  
2. 用 instance-mask 把像素分组：对象1、对象2、背景。  
3. 对每一组分别计算误差（如 RMSE/SILog），得到“按对象的误差表”。  
4. 用 box2d 找出重叠框对（IoU>阈值），把这些区域当“遮挡高风险区域”，单独统计误差。  
5. 最后输出两类指标：  
   - 全局指标（整图）  
   - 实例/遮挡指标（更贴近论文关注点）

这样做的价值：你可以清楚回答“模型到底是哪里坏了”，而不是只看到一个平均分数。

### 用途 C：做“论文思想—通信系统”桥接实验
InstanceDepth 论文强调遮挡关系与实例一致性；你的仓库强调有噪信道下的可训练传输。
GID 是二者共享的数据基础，可用于回答：
- 在同一遮挡场景下，通信失真是否优先破坏前景实例深度一致性？
- 额外发送实例 side information（mask/box）是否能在固定码率下提升深度重建？

以上问题都可在不复现论文网络的前提下，以 GID 作为统一 benchmark 完成对齐研究。

通俗解释：
- 论文那边关心“遮挡时对象深度关系对不对”。
- 你项目关心“经过信道噪声后还能不能把深度传对”。
- 桥接实验就是：在同一批 GID 视频上，同时看“通信质量”与“实例关系质量”。

可以直接做的 3 组对比实验：
1. **无信道噪声 vs 有信道噪声**  
   看噪声会不会先破坏“对象间前后关系”（例如本来前景的人应比后景更近）。
2. **只传 RGB+Depth vs 传 RGB+Depth+实例先验（mask/box）**  
   在相同码率下比较实例区域误差，验证 side information 是否值得传。
3. **固定总码率下的码率重分配**  
   把更多码率分给实例区域（尤其遮挡区域），看是否能更有效提升关键对象深度质量。

这样做的价值：你不必先完整复现论文网络，也能把论文关注的“实例一致性”转化成你系统里的“可量化通信收益”。

## 4. 需要立即注意的工程对齐点（基于当前代码）

### 4.1 `benchmark/data/gid_dataset.py` 与你补充的数据结构存在后缀不一致
当前代码索引 RGB 时只遍历 `*.jpg`，并且 instance mask 路径假设为 `"{stem}-instance.png"`。
你补充的数据是 `images/<video>/<frame>.png` 与 `instance-masks/<video>/<frame>.png`。

因此若直接使用该 loader，会出现：
- 无法发现 PNG RGB 帧；
- 无法命中 mask 文件名（当前代码要求 `-instance` 后缀）。

这不是理论问题，而是“数据命名约定”未对齐问题。

### 4.2 `video-train.txt`/`video-test.txt` 已可直接复用
你提供的数据集切分文件名称与仓库当前约定完全一致（`video-{split}.txt`），这部分是天然对齐的，无需额外转换。

### 4.3 `box2d` 当前尚未接入 loader
现有 `GIDFramePairDataset` 仅输出 RGB/Depth/(可选)Mask，不读取 box2d JSON。
如需复现论文中的实例关系约束（例如成对关系），应在数据层把 box2d 一并输出。

## 5. 与论文数学目标可直接对应的最小实验路径（不引入额外假设）

给出 3 个最小可执行实验（都可在当前仓库范式中实现）：

1. **实例区分的失真评测**  
   用 mask 将深度误差分解为实例内误差与非实例区域误差：
   \[
   \mathrm{MSE}_{inst}=\frac{1}{|\Omega_{inst}|}\sum_{p\in\Omega_{inst}}(\hat d_p-d_p)^2
   \]

2. **遮挡对一致性的影响评测**  
   借助 box2d + mask 建立重叠实例对集合，统计相对深度次序是否被保持。

3. **实例 side-information 的率失真收益**  
   比较“仅 RGB+Depth 传输”与“RGB+Depth+实例先验辅助”在相同 bpp/BPE 下的深度指标差异。

这些实验路径和论文“实例一致性/遮挡关系”目标一致，但不等同于复现论文网络结构本身。

## 6. 可引用的真实链接（论文与数学基础）

- ICCV 2025 论文页面（CVF Open Access）：
  https://openaccess.thecvf.com/content/ICCV2025/html/Liang_Instance-Level_Video_Depth_in_Groups_Beyond_Occlusions_ICCV_2025_paper.html
- ICCV 2025 论文 PDF：
  https://openaccess.thecvf.com/content/ICCV2025/papers/Liang_Instance-Level_Video_Depth_in_Groups_Beyond_Occlusions_ICCV_2025_paper.pdf
- 论文在摘要中给出的代码/数据入口（GID）：
  https://github.com/ViktorLiang/GID
- Scale-invariant depth（Eigen et al., NeurIPS 2014）：
  https://papers.nips.cc/paper/2014/file/91c56ce4a249fae5419b90cba831e303-Paper.pdf
- Mask2Former（CVPR 2022，论文方法中引用）：
  https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html

---

结论（严格按当前可核验信息）：
GID 在你的项目中最有价值的用途，不是“替代你现有 JSCC 模型”，而是作为**真实遮挡群体场景的高质量统一数据基准**，用于评估和驱动“多模态传输系统”向“实例级深度一致性”指标升级。

## 7. 面向你的框架目标：用途 B / 用途 C 在“四大环节”中的作用

以下内容只基于你给出的目标描述，不对未实现代码做推断。

### 7.1 你给出的 4 个目标（重述）
1. 参考外部框架做“可插拔单模态 JSCC 方法集成”，统一到 GID 训练；  
2. 训练时可切换 RGB 图像方法与视频方法，并组合成深度/视频编码路径；  
3. 同时支持“联合压缩传输重建”和“单模态压缩传输重建”两种模式；  
4. 最终形成标准化框架（统一数据处理、统一传输流程、方法即插即用）。

### 7.2 用途 B（实例结构评测）在四大环节的作用

#### 压缩
- 作用：告诉你“哪些区域值得保真”（实例前景/遮挡边界），从而为后续码率分配策略提供依据。  
- 可能效果：同样总码率下，把更多表达能力放在关键实例区域，降低“关键目标深度崩坏”概率。

#### 减载
- 作用：帮助定义“可减载但不伤关键语义”的区域。  
- 可能效果：背景或低关注区域可更激进降载；实例区域保留更多信息，减少任务关键误差。

#### 传输
- 作用：提供“实例敏感”的误差监控视角（不是只看全图）。  
- 可能效果：在相同信道条件下，能更快定位“信道噪声先破坏了哪类对象/哪类区域”。

#### 重建
- 作用：作为重建质量的细粒度验收标准（对象级 + 遮挡级）。  
- 可能效果：即使全局指标接近，仍可区分“实例关系是否被破坏”，更符合 GID/InstanceDepth 场景目标。

一句话总结用途 B：它本质是“评价与诊断层”的升级，让你的标准框架不只会算平均分，而是能定位关键对象质量。

### 7.3 用途 C（论文思想—通信系统桥接）在四大环节的作用

#### 压缩
- 作用：把“实例一致性”目标显式纳入压缩策略比较（不同单模态 JSCC 方法组合的优劣）。  
- 可能效果：同样 bpp/BPE 下，比较哪些方法组合更能保住实例前后层次关系。

#### 减载
- 作用：支撑“按任务价值减载”的实验设计（实例 side-information 是否值得传）。  
- 可能效果：验证“少量 mask/box 先验”是否能抵消部分主信号降载带来的损失。

#### 传输
- 作用：把“信道鲁棒性”与“实例一致性鲁棒性”放到同一实验里做联合对比。  
- 可能效果：得到“哪种方法在噪声下更稳地保持实例深度关系”的证据，而不只是一张 PSNR 表。

#### 重建
- 作用：把重建目标从“像素看起来还行”提升到“实例关系仍正确”。  
- 可能效果：为联合模式 vs 单模态模式提供更有解释力的对比结论。

一句话总结用途 C：它本质是“实验设计层”的升级，把论文关注的实例语义目标转化为你框架里的可量化通信实验。

### 7.4 对应你 4 个目标的直接价值
- 对目标 1（方法集成）：B 提供统一细粒度评测口径；C 提供统一跨方法对比任务。  
- 对目标 2（训练可切换组合）：B/C 都能用于比较“不同 RGB 方法 × 不同视频方法”组合的实例鲁棒性。  
- 对目标 3（联合 vs 单模态模式）：B/C 提供超越全局平均分的对比依据，能判断“为何联合更好/何时不更好”。  
- 对目标 4（标准框架）：B 是标准评测协议的一部分，C 是标准实验协议的一部分，二者共同构成可复用 benchmark。

## 8. 对你最新设想的代码级核查结论（仅基于当前仓库可读代码）

### 8.1 你提出的 5 点设想：逐条判断

1) **“引入 instance-mask 与 box，并加入辅助损失”**  
结论：方向正确，且和 GID/InstanceDepth 目标一致。当前 `benchmark/data/gid_dataset.py` 已有可选 mask 读取入口，但尚未读取 box2d。  

2) **“统一损失与评测指标，避免方法各自为战”**  
结论：方向正确。当前仓库已有统一损失容器 `DepthVideoLoss` 与统一评测入口 `calculate_multimodal_metrics`，可作为标准协议骨架。  

3) **“GID 的连续 images 作为视频模态，删除不合程序”**  
结论：必须做数据层清理。当前 `data_loader.py` 的视频读取路径是 `cv2.VideoCapture(video.file)`（面向视频文件），与 GID 的“帧目录”模式不一致。若你要以 GID 连续帧为视频模态，应把“按视频文件读取”路径改为“按帧序列读取”主路径。  

4) **“共用模块化框架（数据/模型加载/传输/重建/损失评测）”**  
结论：方向正确。当前代码已经有这些模块雏形，但需要把“单模态方法注册 + 模式切换 + GID统一输入输出规范”收敛成同一接口契约。  

5) **“支持只开一模态，与双模态联合做对比”**  
结论：这是必要功能，但当前核心 `DepthVideoJSCC.forward` 明确要求 depth/video 同时输入，不支持单模态关闭。需要在系统编排层加入 mode 开关与旁路逻辑。

---

### 8.2 与 GID“帧序列视频模态”冲突的现有程序（建议收敛/替换）

1. **`data_loader.py` 的视频文件读取路径**  
- 现状：`_load_video_frames` 通过 `cv2.VideoCapture(full_path)` 读取单个视频文件。  
- 与 GID 冲突点：GID 是 `images/<video_name>/00000.png...` 的帧目录，不是单个 mp4。  
- 建议：将该路径降级为“兼容旧数据”，主路径改为“帧目录采样器”。  

2. **`benchmark/data/gid_dataset.py` 的后缀/命名约定**  
- 现状：RGB 只扫 `*.jpg`，mask 假设 `-instance.png` 后缀。  
- 与你提供的 GID 结构冲突：`images/*.png` 与 `instance-masks/<frame>.png`。  
- 建议：改为多后缀兼容（jpg/png）+ mask 文件名直连帧号（优先 `<stem>.png`，再回退 `-instance.png`）。  

3. **主系统单模态关闭能力缺失**  
- 现状：`modules/system.py` 在缺任一模态时直接抛错。  
- 与目标冲突：无法做“单模态开/关”对照实验。  
- 建议：加入 `mode in {joint, depth_only, video_only}`，并在损失与指标侧按 mode 聚合。  

---

### 8.3 你问“重建模块是否和单模态方法保持一致？”

建议是：**编码器/解码器保持“方法原生实现”，系统层只做“接口标准化”**。  
即：
- 单模态 JSCC 方法本身如何压缩-传输-重建，保持其原论文/原实现；
- 你的主框架只强制统一 I/O 契约（输入张量键名、输出重建键名、rate统计键名、可选辅助输入键名）；
- 这样才能做到“公平横向比较”，避免因二次改写破坏方法本体性能。

### 8.4 推荐的统一接口最小集合（用于你的标准框架）

输入（按 batch）：
- `depth_input`（必需于 depth_only/joint）
- `video_input`（必需于 video_only/joint，来自连续 RGB 帧）
- `instance_mask`（可选）
- `box2d`（可选）
- `snr_db`（传输条件）

输出（统一）：
- `depth_decoded`（若存在 depth 路径）
- `video_decoded`（若存在 video 路径）
- `rate_stats`（统一 bpe/bpp 口径）
- `aux_stats`（实例误差、遮挡关系误差、可选 side-info 统计）

统一评测协议：
- 基础 RD：PSNR/SSIM/REL/RMSE/SILog  
- 实例协议：instance-wise 指标 + occlusion 区域指标  
- 模式协议：`joint` vs `depth_only` vs `video_only` 三组并列报告

这样即可同时满足你“框架标准化”和“方法可插拔”两大目标。

## 9. 2020+ 顶会/顶刊参考下的实现蓝图（面向你的最终目标）

本节只给“可执行框架方案 + 数学目标”，不宣称当前仓库已实现。

### 9.1 若要完整接入 GID，你还需补充/确认的数据细节

请补充以下可核验信息（避免后续读取歧义）：
1. `images/<video_name>/` 是否仅含 RGB 帧，且统一尺寸/色彩空间；  
2. `instance-masks/<video_name>/` 的像素编码规则（0 背景？ID 连续？跨帧 ID 是否保持）；  
3. `box2d/<video_name>.json` 的字段定义（`[x1,y1,x2,y2,?,id]` 中第 5 列语义）；  
4. 是否存在无实例帧、空框帧、mask 与 box2d 不一致帧；  
5. 深度值标定：`aligned_depths` 的数值范围是否稳定映射到米制（0.01~10m）。

这些信息决定了 instance/occlusion 损失是否可稳定训练。

### 9.2 标准框架结构（与你目标一一对应）

模块层建议：
1. `DataModule`（统一读取 GID：depth/rgb-sequence/mask/box2d）  
2. `MethodRegistry`（注册单模态 JSCC 方法，按 `depth_method` / `video_method` 实例化）  
3. `TransportModule`（统一信道：AWGN/Rayleigh/Rician + SNR 调度）  
4. `ReconstructionHead`（保持方法原生解码，不做方法内核改写）  
5. `LossMetricSuite`（统一 RD + 实例 + 遮挡 + 模式对比）

该“注册-实例化”思路可借鉴 UnSSR 的 model-zoo 组织方式（其任务不同，但框架组织可参考）。

### 9.3 统一数学目标（joint / single 都可复用）

设模式变量 \(m\in\{\text{joint},\text{depth\_only},\text{video\_only}\}\)，
定义可用模态集合 \(\mathcal{M}(m)\subseteq\{d,v\}\)。

#### (1) 总目标
\[
\mathcal{L}_{total}
=\sum_{k\in\mathcal{M}(m)}\alpha_k\,\mathcal{L}^{rec}_k
+\lambda_R\,\mathcal{R}
+\lambda_{inst}\,\mathcal{L}_{inst}
+\lambda_{occ}\,\mathcal{L}_{occ}.
\]

- \(\mathcal{L}^{rec}_d\)：深度重建（可用 SILog/梯度/法向）  
- \(\mathcal{L}^{rec}_v\)：视频重建（Charbonnier/时序一致）  
- \(\mathcal{R}\)：码率项（bpp/BPE）  
- \(\mathcal{L}_{inst}\)：实例级损失（mask/box2d 驱动）  
- \(\mathcal{L}_{occ}\)：遮挡关系一致性损失

#### (2) 实例级深度误差（可直接由 mask 计算）
\[
\mathcal{L}_{inst}
=\frac{1}{N}\sum_{i=1}^{N}
\frac{1}{|\Omega_i|}
\sum_{p\in\Omega_i}\rho\!\left(\hat d_p-d_p\right),
\]
其中 \(\Omega_i\) 为第 \(i\) 个实例像素集合，\(\rho\) 可取 \(L_1\) 或 Charbonnier。

#### (3) 遮挡关系一致性（由 box2d+mask 建立重叠对）
设重叠实例对集合 \(\mathcal{P}\)，则可定义：
\[
\mathcal{L}_{occ}
=\frac{1}{|\mathcal{P}|}
\sum_{(i,j)\in\mathcal{P}}
\ell\Big(
\operatorname{sgn}(\bar d_i-\bar d_j),\;
\operatorname{sgn}(\bar{\hat d}_i-\bar{\hat d}_j)
\Big),
\]
其中 \(\bar d_i\) 是实例 \(i\) 的平均 GT 深度，\(\bar{\hat d}_i\) 为预测平均深度，\(\ell\) 可用 0/1 排序误差或 hinge。

#### (4) 模式对比协议（你第 5 点目标）
\[
\Delta_{joint-single}
=\operatorname{Metric}_{joint}
-\operatorname{Metric}_{single},
\]
分别在 `depth_only` 与 `video_only` 基线上报告，以检验“双模态联合”是否真实增益。

### 9.4 训练时“方法切换与组合”的统一接口

每次训练给定：
- `depth_method_id`（例如某图像 JSCC 方法）
- `video_method_id`（例如某视频 JSCC 方法）
- `mode`（joint/depth_only/video_only）

系统按 `mode` 自动决定：
- 哪些分支前向；
- 哪些损失项激活；
- 哪些指标写入日志。

这样可以把“方法插件化”与“公平对比”绑定到同一协议。

### 9.5 2020+ 参考文献（真实链接，按模块分组）

#### A) 实例/分割/关系建模
- DETR, ECCV 2020: https://arxiv.org/abs/2005.12872  
- Mask2Former, CVPR 2022: https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html

#### B) 深度估计主干/表示
- DPT (Vision Transformers for Dense Prediction), ICCV 2021:  
  https://openaccess.thecvf.com/content/ICCV2021/html/Ranftl_Vision_Transformers_for_Dense_Prediction_ICCV_2021_paper.html  
- AdaBins, CVPR 2021:  
  https://openaccess.thecvf.com/content/CVPR2021/html/Bhat_AdaBins_Depth_Estimation_Using_Adaptive_Bins_CVPR_2021_paper.html

#### C) 视频表示（可作为视频单模态方法的特征基础）
- TimeSformer, ICML 2021: https://proceedings.mlr.press/v139/bertasius21a.html  
- VideoMAE, NeurIPS 2022: https://arxiv.org/abs/2203.12602

#### D) 压缩与率失真理论（工程必需）
- Variational Image Compression with a Scale Hyperprior, ICLR 2018（RD 基础范式）:  
  https://arxiv.org/abs/1802.01436  
- Shannon 信息论基础（率失真理论根基）:  
  https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf

#### E) 你的目标任务数据与论文
- InstanceDepth (ICCV 2025):  
  https://openaccess.thecvf.com/content/ICCV2025/html/Liang_Instance-Level_Video_Depth_in_Groups_Beyond_Occlusions_ICCV_2025_paper.html  
- GID repository:  
  https://github.com/ViktorLiang/GID

### 9.6 你现在可以直接执行的最小里程碑

M1（数据层）：
- 把 `images/*.png` 帧序列作为视频主输入；
- 接入 `instance-mask` 与 `box2d` 到 batch。

M2（系统层）：
- 加 `mode` 三态开关；
- 加方法注册器（depth/video 分别可插拔）。

M3（训练层）：
- 统一损失为“基础 RD + 实例 + 遮挡”；
- 统一报告 `joint/depth_only/video_only` 三组结果。

M4（对比层）：
- 固定 SNR 与总码率，做“不同方法组合”的公平表格；
- 额外报告实例区与遮挡区指标，不只看全局平均值。

## 10. 基于你新补充信息的“更新修改方案”

### 10.1 已确认事实（用于替换 9.1 的待确认项）

根据你最新回复与仓库样例：
1. 当前 GID 使用 RGB、深度、实例掩码和 box2d；**没有地面掩码**。  
2. `badminton_205641` 的帧 ID 连续 `00000` 到 `00301`。  
3. 不存在无实例帧、空框帧、mask 与 box2d 不一致帧（按你提供信息）。  
4. 样例 `00000` 中，`badminton_205641.json` 的每个条目是 6 元组，且第 6 列与 mask 中实例 ID 对齐（例如 `tid=2/3/4/5` 可在掩码像素中对应命中）。  

### 10.2 `box2d` 字段定义的代码级分析结论（基于样例+论文任务）

对 `badminton_205641.json` 的可见证据：
- 每行固定长度 6：`[x1, y1, x2, y2, c, id]`；  
- 第 5 列 `c` 在该视频中恒为 `0.0`；  
- 第 6 列 `id` 取值在该视频中为 `2..14`；  
- 与样例掩码图对齐时，`id` 对应实例像素可被对应框完整覆盖（实例像素召回为 1.0）。  

因此在工程实现里可采用：
- `x1,y1,x2,y2`：轴对齐框坐标；  
- `id`：实例/轨迹 ID（用于跨帧关联）；  
- `c`：保留字段（当前视频可视为常量，占位或类别位）。  

这一定义与 InstanceDepth 的“实例关系与遮挡建模”用途一致（框用于候选关系，mask 用于像素级监督）。

### 10.3 针对你当前事实的简化实现方案（可直接执行）

#### 数据层（必须先做）
1. `benchmark/data/gid_dataset.py`：
   - RGB 扫描支持 `*.png` 与 `*.jpg`；
   - mask 优先读 `<stem>.png`，其次回退 `<stem>-instance.png`；
   - 新增 `box2d_path` 索引与按帧读取（从 `<video>.json` 取 `<stem>` 键）。
2. `data_loader.py`：
   - 新增“帧目录视频采样器”，输入 `images/<video_name>/`，按 `video_clip_len` 与策略采样连续帧；
   - 保留 `VideoCapture` 路径作为 legacy（便于兼容旧 manifest）。

#### 系统层（实现你的第 5 点）
3. `modules/system.py`：
   - 加 `mode: joint/depth_only/video_only`；
   - `joint` 走现有双分支；
   - `depth_only` 只走 depth 编解码与传输；
   - `video_only` 只走 video 编解码与传输。

#### 损失与评测层（实现你的第 2 点）
4. 统一损失：
   \[
   \mathcal{L}
   =\mathcal{L}_{RD}
   +\lambda_{inst}\mathcal{L}_{inst}
   +\lambda_{occ}\mathcal{L}_{occ}.
   \]
5. 统一评测：
   - 全局 RD 指标；
   - instance-wise 深度指标；
   - occlusion-pair 排序一致性指标；
   - 三模式并列（joint/depth_only/video_only）。

### 10.4 已确认项更新（根据你最新回复）

你已确认：`box2d` 第 5 列 `c` 在全数据集恒为 `0`。  
因此数据层可将 `box2d` 解析固定为：
\[
[x1,y1,x2,y2,\underbrace{0}_{\text{reserved}},id]
\]
并把第 5 列视为保留位，不参与训练损失与评测统计。

这意味着前述实现方案已无关键语义阻塞，可直接进入代码落地阶段（M1→M4）。

## 11. 修改后问题回答（按当前代码事实）

### Q1. 目前模型包括“集成不同单模态 JSCC 方法”的功能吗？

结论分两层：

1) **系统层能力：部分具备**  
`DepthVideoJSCC` 构造函数支持注入自定义 `depth_encoder/depth_decoder/video_encoder/video_decoder/channel/joint_fusion/entropy_model`，因此“替换单模态方法内核”在接口上是支持的。  

2) **训练入口能力：尚未完全产品化**  
当前 `train.py:create_model` 仍固定实例化 `DepthVideoJSCC(...)` 默认组件，没有“按方法ID/配置文件自动注册并切换不同单模态方法”的注册器逻辑。  
也就是说：**可注入，但还不是“开箱即用的方法库切换”**。

补充：本次已加入 `mode in {joint, depth_only, video_only}`，因此单模态与联合模式对照实验可直接运行。

### Q2. 目前模型整体框架（输入→输出）流程梳理

#### Step A: 数据输入层
1. `data_loader.py` 读取样本：
   - 深度：`depth.file` → 单通道张量；  
   - 视频：`video.file` 可是**帧目录**（GID）或**视频文件**（legacy）；  
   - 统一 `collate` 形成 `inputs = {depth_input, video_input, video_frame_mask}`。  
2. `benchmark/data/gid_dataset.py`（基准读取器）可读：
   - RGB: jpg/png  
   - 深度: aligned_depths  
   - mask: `<stem>.png` 或 `<stem>-instance.png`  
   - box2d: `<video>.json`（可选）

#### Step B: 模型编排层（`modules/system.py`）
按 `mode` 分流：
- `joint`：深度编码 + 视频编码 → 融合(shared/private) → 熵估计 → 信道 → 解码；  
- `depth_only`：仅深度编码/信道/解码；  
- `video_only`：仅视频编码/信道/解码。  

#### Step C: 损失与指标层
1. `losses.py::DepthVideoLoss` 汇总：
   - depth 重建项、video 重建项、rate 项、可选 OMIB-like 正则。  
2. `metrics.py::calculate_multimodal_metrics` 统一输出评测指标。  

#### Step D: 输出
模型输出 `ModelForwardOutput`，包括（按模式可能子集）：
- `depth_decoded` / `video_decoded`  
- `rate_stats`  
- `mode`  
-（joint 时）`entropy_stats`、可选 `omib_stats`

这对应你要求的“统一流程 + 模式可切换 + 可做联合/单模态对比”。

## 12. 外部项目 Deep-JSCC-PyTorch 审查结论与接入方案

审查对象：`https://github.com/chunbaobao/Deep-JSCC-PyTorch`

### 12.1 能否“直接”用于 GID 深度图模态编码？

结论：**不能直接无改动接入**，原因如下（基于项目源码）：
1. 其编码器第一层写死 `in_channels=3`（RGB 输入）；  
2. 解码器末层输出 `out_channels=3` + `Sigmoid`（RGB重建）；  
3. 训练脚本数据流以 CIFAR10/ImageNet RGB 图像为目标，损失为图像 MSE。  

因此它是“图像 RGB 单模态 JSCC”实现，不是“单通道深度图 JSCC”实现。若直接把 GID depth（1 通道）喂入会在输入通道维度上不匹配。

### 12.2 论文与数学依据（为什么它可迁移但需适配）

Deep JSCC 的核心是学习映射：
\[
x \xrightarrow{f_\theta} z \xrightarrow{\text{channel}} \tilde z \xrightarrow{g_\phi} \hat x
\]
并最小化失真 \(\mathbb{E}[d(x,\hat x)]\)（常用 MSE 或其变体）。

该思想对“RGB 图像”与“深度图”都成立，因为两者都可视为二维连续信号；差别在于输入统计与损失定义（深度更适配 SILog/尺度相关项）。

Deep-JSCC-PyTorch 代码中还实现了信道输入功率归一化（按潜变量能量归一），对应通信系统中平均功率约束思想。

### 12.3 接入你本地仓库的具体方案（在“不可直接接入”前提下的最小改造）

#### A. 封装为“可插拔单模态方法”
新增适配器类（建议名 `ExternalDeepJSCCAdapter`）：
- 输入：`[B,1,H,W]` depth（可选重复成3通道，或改外部模型首尾通道数）  
- 输出：`depth_decoded`、可选 `rate_stats`（若无熵模型可先置空/占位）  

#### B. 两条可行改造路线
1. **结构改造路线（推荐）**  
   直接把外部模型 `conv1 in_channels=3` 改为 1、末层输出改为 1；  
   这样不引入伪彩色重复，保留深度单通道语义。  
2. **输入适配路线（快速）**  
   depth 复制 3 通道后喂入原模型，再对输出取单通道；  
   工程快，但可能引入冗余耦合，不如结构改造干净。

#### C. 与你当前框架的对接点
1. 在 `modules/system.py` 的 `depth_only` / `joint` 路径中，把 depth 分支替换为适配器实例；  
2. 在 `train.py` 增加 `depth_method_id`（如 `native_cnn`, `deep_jscc_external`）并在 `create_model` 中选择；  
3. 损失层保持统一协议：  
   \[
   \mathcal{L}=\mathcal{L}_{RD}+\lambda_{inst}\mathcal{L}_{inst}+\lambda_{occ}\mathcal{L}_{occ}
   \]
   其中深度重建项可继续用你现有 `DepthLoss`（含 SILog/梯度/法向项）。

#### D. 验证协议（与你目标一致）
固定同一 GID split、同一 SNR、同一有效码率设置，报告：
- `joint` vs `depth_only` vs `video_only`  
- 全局深度指标 + instance-wise + occlusion-pair 指标  
- 对比 `native depth codec` 与 `deep-jscc-adapted depth codec`

### 12.4 参考链接（真实）
- Deep JSCC 原论文（IEEE TWC 2019）：  
  https://ieeexplore.ieee.org/abstract/document/8723589  
- Deep-JSCC-PyTorch 仓库：  
  https://github.com/chunbaobao/Deep-JSCC-PyTorch  
- 神经压缩 RD 范式（Hyperprior）：  
  https://arxiv.org/abs/1802.01436  
- InstanceDepth（ICCV 2025）：  
  https://openaccess.thecvf.com/content/ICCV2025/html/Liang_Instance-Level_Video_Depth_in_Groups_Beyond_Occlusions_ICCV_2025_paper.html

## 13. 按你偏好的“结构改造路线”+UnSSR风格组织的最新修改方案

你明确偏好“结构改造路线”，并希望参考 UnSSR 的方法组织。基于此，建议如下：

### 13.1 目标定义（不变）

你要实现的是：在统一框架中可切换不同单模态 JSCC 方法，并在 GID 上完成
`joint / depth_only / video_only` 可比实验。

### 13.2 结构改造路线（针对 Deep-JSCC-PyTorch）

对外部 DeepJSCC 模型做最小结构改造：
1. 编码器第一层：`in_channels: 3 -> 1`（深度图单通道）；  
2. 解码器最后一层：`out_channels: 3 -> 1`；  
3. 输出激活保持有界（原实现用 Sigmoid），并在训练前统一 depth 标定到固定区间。  

这样做的数学理由是保持同一 JSCC 映射范式，仅改变信号维度：
\[
x_d \xrightarrow{f_{\theta_d}} z_d \xrightarrow{\eta} \tilde z_d \xrightarrow{g_{\phi_d}} \hat x_d,
\]
并优化
\[
\min_{\theta_d,\phi_d}\; \mathbb E[d_d(x_d,\hat x_d)].
\]

### 13.3 UnSSR 风格的方法组织（借鉴“多方法并列可切换”）

参考 UnSSR README 的“支持多方法列表”思路，把本地仓库改成：
- `methods/depth/`：每个单模态 depth-jscc 方法一个适配器文件；  
- `methods/video/`：每个单模态 video-jscc 方法一个适配器文件；  
- `methods/registry.py`：`build_depth_method(id, cfg)` / `build_video_method(id, cfg)`；  
- `train.py` 只接收 `depth_method_id` 与 `video_method_id`，不直接写死类名。

### 13.4 统一训练目标（结构改造后保持一致）

\[
\mathcal{L}_{total}
=\lambda_d \mathcal{L}^{rec}_d
+\lambda_v \mathcal{L}^{rec}_v
+\lambda_R \mathcal{R}
+\lambda_{inst}\mathcal{L}_{inst}
+\lambda_{occ}\mathcal{L}_{occ}.
\]

其中：
- `depth_only`：仅激活 \(\mathcal{L}^{rec}_d\) 与对应 rate；  
- `video_only`：仅激活 \(\mathcal{L}^{rec}_v\) 与对应 rate；  
- `joint`：全部激活。  

### 13.5 代码落地顺序（你现在可以直接执行）

M1（方法侧）：
- 新增 `ExternalDeepJSCCDepth`（结构改造后的 1->1 版本）；
- 保留 `NativeDepthCodec` 作为基线。

M2（注册侧）：
- 新建 registry（depth/video 各一）；
- `train.py` 改为通过 method_id 构造实例。

M3（系统侧）：
- 在 `modules/system.py` 中由 registry 注入 depth/video 分支；
- 保持现有 `mode` 三态与统一输出接口不变。

M4（评测侧）：
- 固定同一 split/SNR/有效码率；
- 比较 `native_depth` vs `deepjscc_depth_adapted`；
- 报告全局 + instance + occlusion 指标。

### 13.6 本节参考链接（真实）
- UnSSR 仓库（方法列表组织参考）：  
  https://github.com/SuperiorLeo/Uncertainty-guided-UnSSR  
- Deep-JSCC-PyTorch：  
  https://github.com/chunbaobao/Deep-JSCC-PyTorch  
- Deep JSCC 论文（IEEE TWC 2019）：  
  https://ieeexplore.ieee.org/abstract/document/8723589  
- Rate-Distortion 神经压缩范式：  
  https://arxiv.org/abs/1802.01436  
- InstanceDepth（ICCV 2025）：  
  https://openaccess.thecvf.com/content/ICCV2025/html/Liang_Instance-Level_Video_Depth_in_Groups_Beyond_Occlusions_ICCV_2025_paper.html
