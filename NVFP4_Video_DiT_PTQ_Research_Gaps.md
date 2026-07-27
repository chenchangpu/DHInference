# 视频生成 DiT 的 NVFP4 W4A4 PTQ：研究空白与创新方向清单

> 整理日期：2026-07-27  
> 研究对象：视频生成模型、Diffusion Transformer、NVFP4、W4A4、PTQ  
> 重点模型：TurboWan2.1-14B、TurboWan2.2-I2V-A14B

## 1. 核心判断

当前遇到的问题并不是“还没有找到合适的 SmoothQuant 参数”，而是已经逐渐接近了**局部张量重构类 PTQ 在 NVFP4 上的收益上限**。

NVFP4 使用：

- 16-element micro-block；
- E2M1 FP4 数据；
- 每 16 个数共享一个 E4M3 FP8 scale；
- tensor-level FP32 二级 scale。

这种细粒度 scaling 本身已经具备较强的局部动态范围适配能力。近期研究进一步指出：

- NVFP4 的 group size 很小，传统 outlier smoothing、rotation 的作用会被显著削弱；
- 对 NVFP4 而言，Hadamard rotation 在部分设置下甚至可能降低精度；
- 主要误差来源不再只是极端 outlier，而是归一化后位于 FP4 大间隔区域的 near-maximal values；
- Four Over Six 因而从 block scale endpoint 入手，在 max-to-4 和 max-to-6 之间选择，但仍然使用 block 局部重构误差作为目标。

参考：

- [Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization](https://arxiv.org/html/2509.23202v3)
- [Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling](https://arxiv.org/html/2512.02010v5)

这些结论能够解释当前实验现象：

- Random Hadamard Rotation、SmoothQuant、SVDQuant 对 NVFP4 没有稳定收益；
- GPTQ 优化单层输出，但最终 bad case 可能由多层、多 step 路径放大决定；
- activation magnitude 随 timestep 下降，却不能准确预测真实敏感性；
- 真正明显的正信号来自 step 和 layer 的非均匀敏感性：
  - step 0 使用 W8A8、后续 step 使用 W4A4，28/28 样本提升，平均约 +2.67 dB；
  - step 0 前部 block 极其敏感；
  - step 0 尾部约 12–20 个 block 可以降为 W4A4，质量损失很小。

因此，后续最值得研究的问题应从：

> 如何继续降低某个 tensor 或 linear layer 的量化 MSE？

转向：

> 哪一个 NVFP4 量化误差会沿 DiT 推理轨迹被放大，并最终造成视频 bad case？

---

## 2. 当前研究版图与空白

| 方向 | 代表工作 | 当前状态 | 仍然存在的空白 |
|---|---|---|---|
| Outlier smoothing / rotation | SmoothQuant、SVDQuant、ViDiT-Q、DiRotQ、KroQuant | 非常拥挤 | NVFP4 block-16 下进一步压制 outlier 的收益有限 |
| Weight Hessian / reconstruction | GPTQ、MR-GPTQ、SVDQuant-GPTQ | 较拥挤 | Hessian 和重构目标通常停留在单层输出，不对应最终视频损失 |
| Timestep / expert aware | DVD-Quant、TS-SVDQ-GPTQ、6Bit-Diffusion | 已经拥挤 | 多数使用 activation 统计或局部误差，未估计真实的下游放大率 |
| Channel permutation / block grouping | PermuQuant | 新兴但已有代表工作 | 尚缺少 step × expert × trajectory influence 联合建模 |
| FP4 scale / grid | 原生 NVFP4、Four Over Six、SemanticDialect | 新兴 | 4/6 scale 仍由局部 MSE 选择，未面向完整扩散轨迹优化 |
| Sampler-side correction | PTQD、Sampling-Aware Quantization、Q-Drift | 部分覆盖 | 多为 timestep 级均值/方差或方向修正，缺少 layer-wise、有方向的 NVFP4 误差建模 |
| Bad-case / tail risk | 相关工作很少 | **明显空白** | 论文普遍优化平均指标，而不直接控制最差 prompt、image condition 和 seed |
| NVFP4-aware distillation for Video DiT | NVIDIA QAD 主要面向 LLM/VLM | **明显空白** | 缺少面向视频时空一致性和完整 denoising trajectory 的 NVFP4 QAD |

### 2.1 近期工作已经覆盖的方向

- [6Bit-Diffusion](https://arxiv.org/html/2603.18742v1)：利用前一 timestep 的 block input-output difference，动态选择 NVFP4 或 INT8。
- [Timestep-Aware SVDQuant-GPTQ for Wan2.2-I2V](https://arxiv.org/html/2605.27003v1)：覆盖双专家、timestep-bin clipping、SVDQuant 和 GPTQ。
- [PermuQuant](https://arxiv.org/html/2605.09503v1)：根据 activation/weight 二阶矩重排 channel，使相近统计特性的 channel 落入同一 quantization group。
- [SemanticDialect](https://arxiv.org/html/2603.02883v3)：研究 semantic-aware mixed format、token correlation 和 residual re-quantization。
- [DiRotQ](https://arxiv.org/html/2605.16732v1)：通过 PCA rotation 和高精度主子空间改善 W4A4。
- [KroQuant](https://arxiv.org/html/2607.21446v1)：使用可映射到 tensor core 的 Kronecker-structured block transform。
- [LongLive-2.0](https://arxiv.org/html/2605.18739v2)：将 NVFP4 用于长视频训练和推理，并采用 Four Over Six scale search。

由此可见，再设计一种普通的 rotation、smoothing 或 timestep clipping，已经较难形成足够强的论文创新。

---

## 3. 最推荐的论文主线

暂定名称：

> **TrajRisk-FP4: Trajectory-Influence and Tail-Risk Aware NVFP4 Quantization for Video Diffusion Transformers**

核心由三部分组成：

1. 证明局部量化误差不能准确预测最终视频伤害；
2. 使用 trajectory influence 优化 NVFP4 scale 和 rounding；
3. 使用 CVaR 或 worst-case objective 显式控制 bad case。

---

## 4. 创新点一：Trajectory Influence 建模

### 4.1 基本假设

将第 \(t\) 个 denoising step、第 \(l\) 个 block 的量化误差记为：

\[
\epsilon_{t,l}
=
h^{q}_{t,l}-h^{fp}_{t,l}.
\]

最终 latent 偏差的一阶近似为：

\[
\Delta z_0
\approx
\sum_{t,l}
J_{(t,l)\rightarrow 0}\epsilon_{t,l},
\]

其中 \(J_{(t,l)\rightarrow0}\) 表示该位置的扰动传播到最终输出的 Jacobian。

现有 GPTQ、SmoothQuant、Four Over Six 等方法主要优化：

\[
\|\epsilon_{t,l}\|^2.
\]

但真正决定最终视频质量的是：

\[
\left\|
\sum_{t,l}
J_{(t,l)\rightarrow0}\epsilon_{t,l}
\right\|^2.
\]

这意味着最终伤害由三部分共同决定：

- 当前量化误差的大小；
- 当前量化误差的方向；
- 误差经过后续 layer 和 timestep 后的放大率。

### 4.2 与现有工作的区别

- GPTQ主要使用当前 linear 输入 Hessian，优化单层 output reconstruction。
- 6Bit-Diffusion 使用前一 timestep 的 block transformation magnitude 预测当前局部量化误差。
- Sampling-Aware Quantization 主要对齐 sampler direction。
- Q-Drift 将量化误差建模为 timestep-wise stochastic perturbation，并修正 sampler drift。

尚未看到公开工作将：

> NVFP4 micro-block scale / rounding × layer × timestep × final-video influence

统一纳入一个 PTQ 目标。

参考：

- [Quantizing Diffusion Models from a Sampling-Aware Perspective](https://arxiv.org/html/2505.02242v1)
- [Q-Drift: Quantization-Aware Drift Correction for Diffusion Model Sampling](https://arxiv.org/abs/2603.18095)

### 4.3 可行的 influence score

对完整 4-step trajectory 做 differentiable fake-quant unroll，从最终损失反传，对每个 block 计算：

\[
S_{t,l}
=
\mathbb E
\left[
\left\langle
\nabla_{h_{t,l}}\mathcal L_{\text{final}},
\epsilon_{t,l}
\right\rangle^2
\right].
\]

其中最终损失可以包含：

- final latent L2 / cosine；
- VAE decode 后的 LPIPS；
- DINO 或 CLIP feature distance；
- 抽样视频帧的 temporal feature consistency；
- optical-flow warping error。

### 4.4 可验证的核心结论

- local MSE 与最终质量下降相关性较低；
- activation range 与最终质量下降相关性较低；
- influence score 与最终质量下降相关性明显更高；
- step 0 前部 block 的 influence 显著高于 step 0 后部；
- bad case 中的误差更容易沿相同方向相干叠加。

---

## 5. 创新点二：Trajectory-aware Four Over Six

### 5.1 原始 Four Over Six

标准 Four Over Six 为每个 NVFP4 block 比较：

\[
\|X-Q_4(X)\|^2
\quad\text{和}\quad
\|X-Q_6(X)\|^2,
\]

然后选择局部重构误差更小的 scale。

### 5.2 改进目标

将 scale 选择改为：

\[
m^*
=
\arg\min_{m\in\{4,6\}}
\mathcal L_{\text{trajectory}}\big(Q_m(X)\big).
\]

创新点不在于增加新的 FP4 格式，而在于：

> 将 NVFP4 scale selection 从 tensor reconstruction 改为 diffusion-trajectory functional optimization。

### 5.3 Weight 侧方案

- 对每个 weight micro-block 生成 max-to-4、max-to-6 两个候选；
- 不按 weight MSE 选择，而按 calibration trajectory 的最终 latent/perceptual loss 选择；
- high-noise expert 和 low-noise expert 分开优化；
- 对共享 weight，聚合它在不同 timestep 的 influence；
- 最终只保存一个标准 NVFP4 weight，不增加 runtime 格式和分支。

### 5.4 Activation 侧方案

在线无法直接计算 trajectory loss，因此可以离线训练一个轻量 policy。

候选输入特征：

- timestep；
- layer id；
- expert id；
- top-1/top-2 magnitude ratio；
- near-max value density；
- block kurtosis；
- saturation ratio；
- max-to-4 与 max-to-6 的局部 disagreement；
- block residual amplification。

输出：

- 选择 max-to-4；
- 或选择 max-to-6。

最终仍然生成标准 NVFP4 activation，不需要设计新的数据格式。

### 5.5 Go / No-Go 判据

- 如果 oracle trajectory-aware 4/6 明显优于 local-MSE 4/6，再实现轻量 policy；
- 如果 oracle 都没有明显提升，说明 scale 选择空间不足，应转向 QAD 或 mixed precision；
- 必须单独测试 weight-only、activation-only 和 W4A4 三种情况。

---

## 6. 创新点三：Bad-case / CVaR-aware PTQ

### 6.1 研究动机

现有量化论文通常优化平均 FID、VBench、PSNR 或 calibration MSE，但实际部署最突出的问题往往是少数 prompt、condition、seed 出现严重崩坏。

因此目标函数可以从均值：

\[
\mathbb E[D]
\]

扩展为：

\[
\mathcal L
=
\mathbb E[D]
+
\lambda\operatorname{CVaR}_{\alpha}(D),
\]

其中 \(D\) 是量化结果相对 BF16 的最终质量损失，CVaR 重点优化最差的 \(5\%\sim10\%\) 样本。

### 6.2 Strict W4A4 模式

- 所有目标 linear 仍使用 NVFP4 W4A4；
- 对高风险样本选择不同的 4/6 scale policy；
- 使用不同的 rounding/clipping policy；
- 不引入 W8A8 fallback。

### 6.3 Budgeted mixed-precision 模式

- 平时运行完整 W4A4；
- 当预测到 bad-case risk 较高时，仅提升 step 0 前部敏感 block；
- 使用固定平均 W8 MAC budget；
- 优化目标是相同平均成本下的 bad-case rate，而不是单纯平均 PSNR。

### 6.4 不依赖 BF16 teacher 的在线风险特征

- step 0 activation saturation rate；
- near-max value比例；
- max-to-4 与 max-to-6 的 self-disagreement；
- 连续 block residual amplification；
- latent energy；
- image condition strength；
- scale entropy；
- 敏感 block 的异常统计。

### 6.5 与 6Bit-Diffusion 的区别

6Bit-Diffusion根据前一 timestep 的 block input-output difference预测局部量化误差，并选择 INT8/NVFP4。

本方向预测的是：

- 最终视频 bad-case risk；
- 尾部质量损失；
- 在固定平均计算预算下，哪些样本和 block 值得提升精度。

---

## 7. 其他创新方向

| 优先级 | 方向 | 核心机制 | 创新潜力 | 主要风险 |
|---|---|---|---|---|
| S | Trajectory-aware NVFP4 scale/rounding | 用最终 latent/video influence 选择4/6 scale和rounding | 高 | calibration反传成本 |
| S | Bad-case CVaR PTQ | 优化最差prompt/seed而非平均MSE | 高 | 需要足够多且足够难的样本 |
| A | Trajectory-aware QAD | NVFP4 student匹配BF16完整denoising trajectory | 高 | 训练资源需求较大 |
| A | Temporal error shaping | 主动让不同step的量化误差方向抵消 | 很高 | kernel与理论难度高 |
| B | Joint FFN reconstruction | 联合优化ffn.0、激活、ffn.2和residual输出 | 中高 | 可能仍然局限于局部收益 |
| B | Influence-aware channel grouping | PermuQuant式重排，但使用trajectory influence目标 | 中 | 与PermuQuant距离较近 |
| C | 更多rotation/SVD/smoothing变体 | 更换transform、rank或clipping | 低 | 研究空间已经拥挤 |

---

## 8. Temporal Error Shaping

### 8.1 核心思想

现有方法通常要求每一步的量化误差都尽可能小，但扩散最终误差取决于各 timestep 误差的加权和：

\[
\Delta z_0
\approx
\sum_t a_t\epsilon_t.
\]

因此不一定需要每个 \(\epsilon_t\) 单独最小，可以主动设计：

\[
\sum_t a_t\epsilon_t\approx0.
\]

即通过控制量化误差方向，使不同 timestep 的误差相互抵消。

### 8.2 可能实现

- activation quantization 在相邻 timestep 使用不同 rounding bias；
- 保存每个 quantization group 的低维误差状态，进行 sigma-delta/error-feedback；
- 对少数最敏感 weight block 准备互补的 NVFP4 rounding 版本；
- 根据 sampler coefficient 选择误差方向；
- 对 step 0 的量化误差进行低维预测，并在后续 step 做补偿。

### 8.3 与现有 sampler correction 的区别

- PTQD主要修正预测均值、相关误差和额外方差；
- Q-Drift主要做 timestep-wise drift correction；
- 本方向直接控制 NVFP4 rounding error 在多 step 上的频谱和相关性。

该方向理论新颖性高，但实现和 kernel 成本也最高，建议作为长期探索方向。

---

## 9. Joint FFN Functional Quantization

当前主要量化 ffn.0 和 ffn.2。与其分别优化：

\[
XW_0,\qquad HW_2,
\]

更合理的目标是联合优化完整 FFN residual update：

\[
X+
Q_A\!\left(
\phi(Q_A(X)Q_W(W_0))
\right)Q_W(W_2).
\]

可联合搜索：

- ffn.0 weight scale/rounding；
- 中间 activation scale；
- nonlinear activation 前后的 clipping；
- ffn.2 weight scale/rounding；
- residual branch output；
- timestep-conditioned AdaLN/residual scale。

潜在收益：

- 让 ffn.0 和 ffn.2 的误差部分抵消；
- 避免分别优化两个 linear，却破坏完整 FFN function；
- 比单层 GPTQ 更符合 DiT block 的真实计算结构。

该方向适合作为 TrajRisk-FP4 的一个子模块，而不建议单独作为整篇论文的唯一创新。

---

## 10. Video NVFP4 Quantization-Aware Distillation

如果 PTQ 已经接近极限，可以转向轻量 training-aware 路线。

### 10.1 基本方案

- BF16 模型作为 teacher；
- NVFP4 fake-quant 模型作为 student；
- 冻结绝大多数主干参数；
- 只训练少量参数：
  - AdaLN scale；
  - 低 rank LoRA；
  - clipping/rounding参数；
  - scale policy；
  - 小型低 rank error correction。

### 10.2 Loss设计

- noise/velocity prediction alignment；
- intermediate block feature alignment；
- full latent trajectory alignment；
- final latent alignment；
- temporal feature consistency；
- CVaR bad-case loss；
- 对 step 0 前部敏感 block 增加权重。

### 10.3 部署约束

- correction最终尽量 merge回weight；
- merge后重新量化为标准NVFP4；
- 避免保留高精度低rank分支；
- 如果必须保留低rank分支，应进一步量化该分支。

NVIDIA 已在 LLM/VLM 上证明 QAD 可以显著恢复 NVFP4 精度，但面向 Video DiT 完整 denoising trajectory 的 QAD 仍然缺少系统研究：

- [Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf)

---

## 11. 判定性实验清单

### 11.1 验证局部指标是否失效

- [ ] 记录每个 step × block 的 activation rel-L2。
- [ ] 记录每个 linear/block 的 output MSE 和 cosine similarity。
- [ ] 单独量化指定 block，继续运行剩余 trajectory。
- [ ] 记录最终 latent PSNR、SSIM、LPIPS 和 feature distance。
- [ ] 计算 local MSE 与最终质量下降的 Spearman 相关性。
- [ ] 计算 activation magnitude 与最终质量下降的相关性。
- [ ] 计算 6Bit-Diffusion 的 \(\Gamma=\|Y-X\|_1/\|X\|_1\) 与最终损失的相关性。
- [ ] 计算 proposed influence score 与最终质量下降的相关性。
- [ ] 分别对 good case 和 bad case 做上述统计。

如果 local MSE 相关性低，而 influence score 相关性明显更高，这可以成为论文的核心观察。

### 11.2 分析误差大小与误差方向

- [ ] 将量化误差投影到 BF16 block update 方向。
- [ ] 分析与 BF16 update 平行和正交的误差分量。
- [ ] 构造 MSE 接近但方向不同的扰动。
- [ ] 比较两类扰动对最终视频的影响。
- [ ] 计算不同 block 量化误差之间的 cosine。
- [ ] 判断多 block 误差是相加、抵消还是出现非线性交互。
- [ ] 比较 bad case 与 good case 的 coherent error accumulation。

### 11.3 测试 trajectory-aware 4/6 上限

- [ ] Native NVFP4 max-to-6。
- [ ] Local-MSE Four Over Six。
- [ ] Layer-wise trajectory-aware 4/6 oracle。
- [ ] Micro-block trajectory-aware 4/6 oracle。
- [ ] Learned lightweight scale policy。
- [ ] Weight-only trajectory-aware 4/6。
- [ ] Activation-only trajectory-aware 4/6。
- [ ] Weight + activation trajectory-aware 4/6。
- [ ] 分别对 high-noise/low-noise expert评估。

### 11.4 Mixed-precision公平比较

- [ ] Pure W4A4。
- [ ] Step 0 W8A8 + later W4A4。
- [ ] Step 0前N个block W8A8。
- [ ] 固定 layer-wise W8 allocation。
- [ ] 6Bit式动态routing。
- [ ] Influence-aware固定routing。
- [ ] Risk-aware sample-conditional routing。
- [ ] 所有方法使用相同 W8 MAC比例或相同真实延迟预算。

### 11.5 Bad-case评估

- [ ] 平均 PSNR/SSIM/LPIPS。
- [ ] 5th percentile PSNR/SSIM。
- [ ] 95th percentile LPIPS。
- [ ] bad-case rate。
- [ ] temporal warping error。
- [ ] subject consistency。
- [ ] background consistency。
- [ ] motion smoothness。
- [ ] 不同 prompt 类型的分组结果。
- [ ] 不同 image condition 强度的分组结果。
- [ ] 不同 motion intensity 的分组结果。
- [ ] 多 seed 稳定性。

### 11.6 系统评估

- [ ] 使用真实 NVFP4 kernel，而不只做 fake quantization。
- [ ] 测量量化和scale selection开销。
- [ ] 测量单个linear latency。
- [ ] 测量单step latency。
- [ ] 测量端到端视频生成延迟。
- [ ] 测量显存占用。
- [ ] 统计 W8 fallback比例。
- [ ] 检查动态policy是否破坏CUDA Graph。
- [ ] 检查scale policy是否能融合进quantization kernel。
- [ ] 在B200/RTX 5090等Blackwell设备上验证实际加速。

---

## 12. 建议暂时停止投入的方向

- [ ] 不再大规模搜索 Random Hadamard size、sign和排列。
- [ ] 不再只搜索单一 SmoothQuant \(\alpha\)。
- [ ] 不再只按 timestep activation max/mean 分 bin。
- [ ] 不再单纯对 SVD rank、GPTQ block size 做网格搜索。
- [ ] 不再只优化 per-layer output MSE。
- [ ] 不优先设计无法映射 Blackwell NVFP4 Tensor Core 的新4-bit格式。
- [ ] 不只在少量样本上报告平均PSNR。
- [ ] 不忽略 bad-case rate 和尾部指标。
- [ ] 不将 fake quantization 的收益直接等同于实际部署收益。

这些方向并非完全无效，但已经难以形成：

> 为什么已有方法在 NVFP4 Video DiT 上失效，以及新方法为什么有效

这一强论文叙事。

---

## 13. 推荐落地顺序

### 阶段一：验证核心假设

1. 完成 local metric 与 final damage 的相关性实验。
2. 构建 step × layer × expert trajectory influence heatmap。
3. 验证 step 0 前部敏感性是否能由 influence score解释。
4. 验证 good case 与 bad case 的误差传播差异。

### 阶段二：确定算法上限

5. 实现 local-MSE Four Over Six baseline。
6. 实现 oracle trajectory-aware 4/6。
7. 比较 oracle 与普通 4/6 的收益差。
8. 如果 oracle 收益明显，继续训练轻量 scale policy。
9. 如果 oracle 收益很小，转向 QAD 或 risk-aware mixed precision。

### 阶段三：解决 bad case

10. 建立更大的 prompt × image × seed calibration/evaluation 集。
11. 使用 mean + CVaR 目标优化。
12. 构建 runtime bad-case risk predictor。
13. 比较 strict W4A4 policy 与 budgeted W8 fallback。

### 阶段四：形成完整论文

14. 加入 joint FFN functional reconstruction 作为增强模块。
15. 在必要时加入 lightweight trajectory-aware QAD。
16. 完成真实NVFP4 kernel和端到端性能评估。
17. 在 Wan2.1/Wan2.2 之外增加至少一个 Video DiT 验证泛化性。

---

## 14. 最终推荐

如果只能选择一个方向，优先选择：

> **Trajectory-influence-weighted NVFP4 scale/rounding + CVaR bad-case objective**

它具备以下优势：

- 能够直接解释当前观察到的 step 0 和层级敏感性；
- 避开已经拥挤的 smoothing、rotation、SVD 和普通 timestep-aware 路线；
- 可以保持标准 NVFP4 数据格式；
- 同时兼容 strict W4A4 和 mixed-precision 部署；
- 研究目标从平均局部误差提升到最终视频质量和 bad-case risk；
- 容易形成“现象分析—理论建模—算法设计—真实系统验证”的完整论文结构。

---

## 15. 主要参考文献

1. [Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization](https://arxiv.org/html/2509.23202v3)
2. [Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling](https://arxiv.org/html/2512.02010v5)
3. [ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers](https://github.com/thu-nics/ViDiT-Q)
4. [DVD-Quant: Data-free Video Diffusion Transformers Quantization](https://arxiv.org/html/2505.18663v4)
5. [Q-VDiT: Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers](https://proceedings.mlr.press/v267/feng25q.html)
6. [Timestep-Aware SVDQuant-GPTQ for W4A4 Quantization of Wan2.2-I2V](https://arxiv.org/html/2605.27003v1)
7. [6Bit-Diffusion: Inference-Time Mixed-Precision Quantization for Video Diffusion Models](https://arxiv.org/html/2603.18742v1)
8. [PermuQuant: Lowering Per-Group Quantization Error by Reordering Channels for Diffusion Models](https://arxiv.org/html/2605.09503v1)
9. [SemanticDialect: Semantic-Aware Mixed-Format Quantization for Video Diffusion Transformers](https://arxiv.org/html/2603.02883v3)
10. [DiRotQ: Rotation-Aware Quantization for 4-bit Diffusion Transformers](https://arxiv.org/html/2605.16732v1)
11. [KroQuant: Kronecker-Structured Block Transforms for Efficient PTQ of DiTs](https://arxiv.org/html/2607.21446v1)
12. [Quantizing Diffusion Models from a Sampling-Aware Perspective](https://arxiv.org/html/2505.02242v1)
13. [Q-Drift: Quantization-Aware Drift Correction for Diffusion Model Sampling](https://arxiv.org/abs/2603.18095)
14. [LongLive-2.0: An NVFP4 Parallel Infrastructure for Long Video Generation](https://arxiv.org/html/2605.18739v2)
15. [Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf)
16. [RFC: NVFP4 Quantization Support for Diffusion Models](https://github.com/vllm-project/vllm-omni/issues/1959)
