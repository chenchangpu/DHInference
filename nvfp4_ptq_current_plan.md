# DiT 视频生成模型 NVFP4 PTQ：当前精度优化方案

> 更新日期：2026-08-05  
> 研究对象：TurboDiffusion 4-step 视频 DiT，Linear 与 Sparse Attention 的 NVFP4 PTQ  
> 约束：尽量不备份完整/部分 W8 主权重；优先保持统一 NVFP4 Tensor Core 路径；推理延迟增量尽可能小。

## 1. 当前问题

### 1.1 Linear 全程 NVFP4 精度显著下降

当前观察：

- 4 个 denoising step 中，第 1 步使用 W8A8、后 3 步使用 NVFP4，生成效果较好；
- 4 个 step 全部使用 NVFP4 时，生成质量明显下降；
- 直接保留第 1 步 W8A8 会引入完整或部分 W8 权重副本，削弱 NVFP4 的显存收益。

核心判断：第 1 个 denoising step 是少步生成轨迹的“锚点”。其量化误差不仅造成当前层输出偏差，还会改变后续 3 个 step 的输入分布，因此误差会沿生成轨迹非线性放大。问题不能只用单层 weight/activation MSE 描述，而应优化最终 trajectory distortion。

### 1.2 SageAttention3 迁移到 Sparse Attention 后损失较大

当前观察：

- SageAttention3 用于 dense attention 时效果较好；
- 将相同量化策略迁移到 TurboDiffusion sparse attention 后，精度损失显著增大。

核心判断：Sparse attention 中存在两种额外的非连续误差放大：

1. **Support/route flip**：量化误差改变 Top-K 边界，导致整个 tile 被误删或误选；
2. **稀疏 softmax 质量偏移**：被保留的 tile 本身通常具有较大 attention mass，其 QK、P/V 量化误差对最终输出更敏感，同时量化后的分子和归一化分母可能不一致。

因此，attention 优化需要同时保护 sparse support、softmax probability mass 和重要 tile 的数值精度。

---

## 2. 首轮定位实验

在实现复杂补偿前，先判断主要误差来源，以避免不必要的 W8 权重备份或高精度计算路径。

### 2.1 Linear：第 1 步 W/A 误差 2×2 分解

后 3 个 step 固定使用 W4A4，仅修改第 1 步：

| 第 1 步配置 | 目的 |
|---|---|
| W8A8 | 高精度参考 |
| W4A8 | 隔离 weight FP4 误差 |
| W8A4 | 隔离 activation FP4 误差 |
| W4A4 | 测量 weight/activation 联合误差 |

结果解释：

- 若 W4A8 接近 W8A8：主要问题来自 activation，无需备份 W8 权重；
- 若 W8A4 接近 W8A8：主要问题来自 weight code/scale；
- 若 W4A4 的损失明显大于 W4A8 与 W8A4 损失之和：存在显著的 \(\Delta X\Delta W\) 交互，需要联合残差补偿。

除最终视频质量外，应记录各 layer/block 的输出 cosine、相对 RMSE，以及每个 step 的 latent drift，确认首步误差从何处开始放大。

### 2.2 Sparse Attention：route、QK、PV 分路测试

固定同一组 Q/K/V，比较：

1. BF16 router/mask + BF16 attention；
2. BF16 mask + NVFP4 QK；
3. BF16 mask + NVFP4 PV；
4. NVFP4 router/mask + BF16 tile compute；
5. 全 NVFP4 sparse attention。

重点指标：

- sparse mask Jaccard；
- BF16 softmax probability mass recall；
- Top-K 边界 margin 与 route flip 比例；
- 每行 softmax mass/normalizer 偏差；
- sparse branch 与 linear branch 的输出范数比；
- attention output cosine、RMSE；
- 最终视频质量与时序一致性指标。

---

## 3. Linear NVFP4 PTQ 候选方案

### L1. Trajectory-aware shared-code dual-scale

#### 核心思想

不再按普通 weight MSE 或静态 absmax 规则选择 NVFP4 scale，而是利用真实 rCM 4-step teacher trajectory，使 scale 直接面向最终生成轨迹优化。

对于 weight group \(g\)，使用近似 Hessian/Fisher 加权目标：

\[
\mathcal L_g =
\sum_t \omega_t\,
\operatorname{Tr}\!\left[
(W_g-\hat W_g^{(t)})H_{t,g}
(W_g-\hat W_g^{(t)})^\top
\right],
\]

其中：

- \(H_{t,g}=X_{t,g}^{\top}X_{t,g}\)，由真实 4-step student activation 估计；
- \(\omega_t\) 表示 timestep 对最终 latent/video 的敏感度；
- 首步通常具有更高权重，但权重应通过一次 backward、Hutchinson 估计或有限差分获得，而不是完全手工指定。

> 实现时应使用维度一致的二次型；对于常见 linear 权重布局，可等价写为 \(\operatorname{Tr}[(W_g-\hat W_g)H_{t,g}(W_g-\hat W_g)^\top]\)。

#### 两级实现

**L1-a：单 scale、零额外推理存储**

- 每组仍只有一套 FP4 code 和 E4M3 scale；
- 从 `max/4`、`max/6` 及相邻可表示 E4M3 scale 中搜索；
- 用 trajectory-aware objective 代替普通 weight MSE 选解；
- 推理 kernel 与数据布局不变。

**L1-b：共享 FP4 code、双 scale view**

- E2M1 weight code 完全共享；
- 仅为敏感 group 保存 \(s_g^{(0)}\) 与 \(s_g^{(1+)}\) 两个 scale；
- step 0 切换到首步 scale，其余 step 使用通用 scale；
- GEMM 数量、输入 dtype 和 Tensor Core 路径保持不变。

#### 存储与性能

NVFP4 每 16 个值约为 \(16\times4+8=72\) bit。每组额外增加一个 8-bit scale：

- 全量双 scale：约增加 11.1% NVFP4 weight-group 存储；
- 仅 20% 敏感 group 使用双 scale：总权重开销约增加 2.2%；
- 推理计算量基本不变，仅增加 step-dependent scale pointer/offset 选择。

#### 价值与风险

- 优点：最符合“不备份 W8、统一 FP4 GEMM、低延迟”的要求，应作为 Linear 第一优先级；
- 风险：共享 code 可能限制两套 scale 的最优性，需要评估 code 固定后 scale 切换是否足以覆盖 step 0 分布。

### L2. First-step Fused Residual-K Lifting

#### 核心思想

若 L1 仍不足以恢复首步精度，不引入独立 W8/FP16 residual GEMM，而是把少量 NVFP4 residual 拼接到原 GEMM 的 reduction 维，一次 FP4 GEMM 完成主路径与补偿。

设：

\[
\bar W=Q_4(W),\qquad E=W-\bar W.
\]

根据首步 activation Hessian/Fisher，从 K 维选择少量敏感、且满足 16 对齐的 channel group \(S\)，仅保存：

\[
\bar E_S=Q_4(E[:,S]).
\]

首步计算改为：

\[
Y_0 \approx
[Q_4(X_0),\ Q_4(X_{0,S})]
[\bar W,\ \bar E_S]^\top.
\]

本质是将 K 维从 \(K\) 扩展到 \(K+|S|\)，仍执行一次标准 NVFP4 GEMM。后 3 步只使用原始 K，不读取 residual segment。

若定位实验显示 activation 和 weight 均敏感，可进一步使用：

\[
Y_0\approx
QX\,\bar W+
QX_S\,\bar E_S+
Q(R_X)_S\,\bar W_S,
\]

并忽略较小的二阶项 \(R_XE\)。三个 FP4 项可统一拼入 reduction 维。

#### 预算估计

若选择 3% channel：

- weight-only 补偿：首步 GEMM 约增加 3%，4-step 平均约增加 0.75%；
- weight + activation 补偿：首步约增加 6%，4-step 平均约增加 1.5%；
- 额外权重存储约为原 NVFP4 payload 的 3%～6%；
- 无 W8 权重副本，无独立高精度 GEMM。

#### 关键实现点

- channel group 必须与 NVFP4 group 和 GEMM K-tile 对齐；
- 敏感度以首步 trajectory loss 为目标，而不是只按 residual absmax 排序；
- residual 段仅在 step 0 参与 K-loop，需避免影响后续 step 的 kernel shape/cache 策略；
- 优先验证 1%、2%、3%、5% channel budget 的质量—延迟 Pareto 曲线。

---

## 4. Sparse Attention NVFP4 PTQ 候选方案

### A1. Margin-Guarded Support-Stable Routing

#### 核心思想

先保护 sparse mask 的稳定性，再优化 active tile 内部的 NVFP4 数值。对于 coarse router score：

\[
r_{ij}=\bar q_i\bar k_j^\top/\sqrt d,
\]

根据 Q/K 量化残差构造误差上界：

\[
\epsilon_{ij}\lesssim
\frac{
\lVert e_{q_i}\rVert\lVert k_j\rVert+
\lVert q_i\rVert\lVert e_{k_j}\rVert+
\lVert e_{q_i}\rVert\lVert e_{k_j}\rVert
}{\sqrt d}.
\]

若第 \(k\) 与第 \(k+1\) 个 router score 的 margin 大于相应误差界，则 mask 可视为稳定；否则将其视为 ambiguous boundary。

#### 实现选项

按改动量由小到大：

1. coarse router/mask 保留 BF16 或 FP8，active sparse tile 继续使用 NVFP4；
2. router 先低精度计算，只对 ambiguous boundary 候选重算 BF16；
3. 不重算，而是将 guard band 内的边界 tile 一并加入 sparse mask，降低误删概率；
4. guard band 按 layer/head/timestep 校准，step 0 使用更保守阈值。

#### 预期收益与开销

- router 通常基于 pooled block，成本远低于 token-level sparse attention；
- 少量边界重算或冗余 tile，预计比将 active attention 全部切到 FP8/BF16 更便宜；
- 主要收益应体现在 mask Jaccard、BF16 mass recall 与最终时序一致性的恢复。

#### 风险

- 误差上界过松会选入过多 tile，降低 sparsity；
- 应使用校准数据统计实际 score error quantile，对理论上界做收紧；
- 若 BF16 mask + NVFP4 QK/PV 已明显变差，则 A1 只能解决 route flip，仍需结合 A2/A3。

### A2. Mass-Consistent Sparse Softmax + Tile Residual Moment

#### A2-a：量化概率一致归一化（PNQ 思路）

普通在线 softmax 可能使用未量化 exponential 更新 normalizer，却使用量化后的 exponential 执行 PV，造成分子与分母不一致。改为：

\[
\hat w_{ij}=Q_{\mathrm{NVFP4}}\!\left(\exp(S_{ij}-m_i)\right),
\]

并让 normalizer 与 PV 共同使用 \(\hat w\)：

\[
Z_i=\sum_j\hat w_{ij},\qquad
N_i=\sum_j\hat w_{ij}V_j,\qquad
O_i=N_i/Z_i.
\]

该方案不增加 MMA，只改变 reduction 的数据源与累加逻辑，应优先作为低开销 baseline。

#### A2-b：Active tile 零阶残差矩修正

对每个 active KV tile \(t\)，保留量化前后 attention weight 的 mass residual：

\[
\delta_{it}=\sum_{j\in t}(w_{ij}-\hat w_{ij}),\qquad
\mu_{V,t}=\frac{1}{B_K}\sum_{j\in t}V_j.
\]

输出修正为：

\[
O_i\approx
\frac{
\sum_t\hat w_{it}V_t+
\sum_t\delta_{it}\mu_{V,t}
}{
\sum_t\hat w_{it}+
\sum_t\delta_{it}
}.
\]

该修正对 tile 内 V 的常量分量是精确的，剩余误差只与 \(V_j-\mu_{V,t}\) 有关。

#### 预算估计

- 每个 active tile 增加一次近似 \(B_Q\times D\) 的修正；
- 相对 \(B_QB_KD\) 的 PV 主计算，理论算术增量约为 \(1/B_K\)；
- 当 \(B_K=64\) 时约为 1.6%；
- \(\mu_V\) 可按 KV tile 预计算，并在同一 tile 被多个 Q block 使用时复用；
- 若 tile mean 不够，可改为每 16-key group 的 correction，但理论开销会上升到约 6.25%。

#### 风险

- 如果 V 在 tile 内变化很大，零阶矩不足，需要验证一阶/分组修正是否值得；
- 应区分误差来自 P 量化还是 V 量化，只有 P/mass 主导时该方案收益最明显；
- 需要与 sparse online softmax 的 max rescale 逻辑严格对齐，避免跨 tile 的 \(\delta\) 定义不一致。

### A3. Hot-in-Hot Tile Dual-Stage NVFP4

#### 核心思想

Sparse routing 已从 dense tiles 中选出约 10%～15% 的重要 tile；再从 active tiles 中选择约 10% 的 hot core，即只对总 dense tiles 的约 1%～1.5% 做额外全 FP4 残差计算，而不是切换到 FP16。

tile 重要度使用输出误差代理，而非只看 attention score：

\[
I_{ij}\approx
\operatorname{mass}_{ij}\cdot
\widehat\epsilon_{ij}\cdot
\lVert V_j-O_i\rVert_{\mathrm{RMS}}.
\]

对 hot-core tile，分解：

\[
Q=Q_4+R_Q,\qquad K=K_4+R_K,
\]

使用统一 NVFP4 MMA 计算：

\[
QK^\top\approx
Q_4K_4^\top+
Q_4R_K^\top+
R_QK_4^\top,
\]

忽略二阶项 \(R_QR_K^\top\)。

#### 使用条件

- A1 已使 sparse mask 基本稳定，但 BF16 mask + NVFP4 QK 仍有明显误差；
- 误差集中于少量高 mass、高量化残差 tile；
- 可只在 step 0 和少数敏感 layer/head 启用。

#### 优势与风险

- 全路径保持 FP4 Tensor Core，不引入异构 FP16 tile path；
- 额外 MMA 只覆盖约 1%～1.5% dense tiles；
- kernel 调度与 residual Q/K 的生成、缓存可能带来非算术开销，需要端到端 profile；
- 二阶 residual 在极端 outlier tile 上可能不可忽略，应先测交叉项误差占比。

---

## 5. 当前推荐落地顺序

| 阶段 | 实验/方案 | 目标 | 预期额外延迟 |
|---|---|---|---:|
| 0 | Linear 2×2 + Attention 五路拆分 | 确认首要误差源 | 仅离线实验 |
| 1 | 真实 4-step calibration + L1-a trajectory-aware scale search | 无运行时改动的基础提升 | 近似 0 |
| 2 | L1-b shared-code dual-scale | 修复 step 0 且不备份 W8 | 近似 0 |
| 3 | A1 support-stable router + A2-a mass-consistent softmax | 修复 sparse route 与 row-mass bias | 很小 |
| 4 | A2-b tile residual moment | 修复 active tile 的 P/V 输出偏差 | PV 约 1%～3% |
| 5 | L2 residual-K / A3 hot-tile dual-FP4 | 对残余敏感区域做定向补偿 | 按预算控制 |

推荐的论文级组合：

> **Linear：Trajectory-aware shared-code dual-scale + first-step residual-K lifting**  
> **Attention：Support-stable routing + mass/moment-conserving sparse NVFP4**

共同原则：不以张量自身的 FP4 MSE 为唯一目标，而是分别保护 diffusion trajectory、sparse support 和 attention probability mass，同时尽量保持统一 NVFP4 GEMM/MMA 路径。

---

## 6. 建议的实验矩阵

### 6.1 Linear ablation

| 实验 | 变量 | 关键结论 |
|---|---|---|
| W/A 2×2 | W8A8、W4A8、W8A4、W4A4 | weight、activation、交互项占比 |
| Scale objective | absmax、局部 MSE、activation MSE、trajectory loss | trajectory calibration 是否必要 |
| Scale candidates | max/4、max/6、相邻 E4M3 scale | ScaleSearch 收益来源 |
| Dual-scale coverage | 5%、10%、20%、50%、100% group | 存储—质量 Pareto |
| Residual-K budget | 1%、2%、3%、5% channel | 延迟—质量 Pareto |
| Residual target | weight-only、activation-only、joint | 是否需要联合补偿 |

### 6.2 Sparse Attention ablation

| 实验 | 变量 | 关键结论 |
|---|---|---|
| Mask precision | BF16、FP8、NVFP4 | route flip 是否为主因 |
| Guard strategy | 无、边界重算、扩展 mask | 精度—稀疏率 Pareto |
| Softmax reduction | 原逻辑、量化 P 一致归一化 | row-mass bias 占比 |
| Moment granularity | 无、per-tile、per-16-key | correction 精度—开销 |
| Hot-core ratio | 0%、5%、10%、20% active tiles | A3 收益是否集中 |
| 启用范围 | 全 step、step 0、敏感 layer/head | timestep/layer 动态性 |

### 6.3 统一评价指标

- 端到端视频质量：使用当前项目既有主指标，并补充时序一致性；
- trajectory：各 step latent cosine、相对 RMSE、最终 latent drift；
- linear：layer output error、channel/group sensitivity 分布；
- attention：mask Jaccard、mass recall、route flip、row sum/normalizer error、output cosine/RMSE；
- 系统：显存、weight payload、首步/平均 step latency、端到端 latency、kernel 数量、Tensor Core 利用率。

---

## 7. 已排除方案

### L3. NVFP4 Group Isolation

状态：**已实验，效果不好，排除。**

原始思路是通过离线 channel permutation，将 step-0 outlier 集中到少量 16-element group，再只对这些 group 做更精细的 scale search 或 residual 补偿。实际实验未获得足够收益，因此：

- 不再列入当前候选方案；
- 不进入后续优先级和主 ablation；
- 后续只有在新证据表明现有实现的 permutation 位置、折叠方式或分组约束存在明显问题时，才考虑重新开启。

---

## 8. 相关工作

- [TurboDiffusion](https://arxiv.org/html/2512.16093v1)：4-step 视频扩散与 sparse attention 实验背景。
- [SageAttention3](https://arxiv.org/html/2505.11594v3)：FP4 attention 量化、Q/K smoothing 等设计。
- [6Bit-Diffusion](https://arxiv.org/html/2603.18742v1)、[AdaTSQ](https://arxiv.org/abs/2602.09883)：Video DiT 随 timestep 变化的量化敏感度。
- [Four Over Six](https://arxiv.org/html/2512.02010v5)、[ScaleSearch](https://arxiv.org/html/2605.12464v1)：NVFP4 E2M1 code/scale 选择与误差优化。
- [ARCQuant](https://arxiv.org/html/2601.07475v1)、[SharQ](https://arxiv.org/html/2606.26587v1)：统一低精度路径中的残差补偿思路。
- [ThriftAttention](https://arxiv.org/html/2605.23081v1)：高 attention-weight tile 的功能性量化敏感度。
- [MXAttention](https://arxiv.org/html/2607.24377v1)：量化概率与 softmax normalizer 一致化。
- [Q-Drift](https://arxiv.org/html/2603.18095v1)、[Q-Sched](https://arxiv.org/html/2509.01624v1)：可作为后续 sampler-side 低开销轨迹修正参考。

## 9. 当前决策摘要

1. Linear 首选 L1，先做 trajectory-aware scale search，再验证 shared-code dual-scale；
2. L1 不足时，使用 L2 以少量 FP4 residual-K 修复首步，不保存 W8 主权重；
3. Sparse attention 先用 A1 判断并稳定 support，再用 A2 保证 softmax mass 一致；
4. 若误差仍集中在少数高价值 tile，再启用 A3 的 hot-in-hot 全 FP4 残差路径；
5. L3 NVFP4 group isolation 已验证效果不佳，正式排除。
