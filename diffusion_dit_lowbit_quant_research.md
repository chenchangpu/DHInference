# Video Diffusion Transformer MLP 低比特量化研究笔记与论文思路

> 研究对象：视频生成 Diffusion Transformer / DiT，重点关注 Wan2.1-14B 中 MLP / FFN 线性层的 W4A4、W4A8 低比特量化。  
> 目标：从现有 LLM 低比特量化、Diffusion 量化、Video DiT 量化文献出发，提出可发展为高质量论文的创新算法方向。

---

## 1. 背景与核心问题

Wan2.1-14B 属于大规模视频生成 DiT 模型，其 DiT 主干包含大规模线性层和 FFN/MLP 模块。公开资料显示，Wan2.1 14B 版本 DiT hidden dimension 为 5120，FFN dimension 为 13824，包含 40 heads 和 40 layers；此外，时间 embedding 通过共享 MLP 产生 modulation 参数，用于调节 transformer block 的行为 [R1][R2]。这意味着 MLP/FFN 线性层既是主要参数与算力开销来源，也是量化误差在多层、多 timestep 推理路径上累积的重要入口。

已有 DiT / Video DiT 量化工作普遍指出，Diffusion Transformer 与 LLM 的量化难点不同：它不仅存在 channel outlier，还存在显著的 timestep-dependent activation distribution、token variance、spatial-temporal variance 和 calibration variance [R3][R4][R5][R6][R7]。因此，在 Wan2.1-14B 上直接套用 LLM 中常见的 SmoothQuant、AWQ、GPTQ、Hadamard rotation、SVDQuant 等方法，即使平均指标可接受，也容易出现 bad case。

本笔记的核心判断是：

**W4 weight 或 W4A4 量化导致的 bad case 不只是单层 reconstruction error 问题，而是 Video DiT denoising trajectory 中“层 × timestep × 视频 token”误差传播和放大的结果。**

因此，新的算法设计应从以下三个维度突破：

1. **Trajectory-aware**：不只优化当前层局部误差，而要估计量化误差对后续 denoising 轨迹的放大效应。
2. **Timestep-conditioned**：不使用全局静态 clip、scale、rotation 或 low-rank compensation，而是按 denoising phase / timestep bin 自适应。
3. **Video-token-aware**：不只按 channel 或 layer 处理 outlier，还要识别 motion-salient、attention-salient、foreground-salient token。

---

## 2. 相关工作梳理

### 2.1 LLM 低比特量化方法

#### GPTQ

GPTQ 是经典的一次性 post-training weight quantization 方法，使用近似二阶信息对 transformer 权重进行 3/4 bit 量化，目标是尽量降低线性层输出重构误差 [R8]。它对 LLM 权重量化非常有效，但其优化对象主要是静态 token 分布下的局部 layer reconstruction error。

对 Video DiT 的启示：

- 可作为 MLP weight residual quantization 的基础；
- 但单纯 GPTQ 无法处理 denoising timestep 变化和视频时序一致性；
- 在 W4A4 场景下，activation quantization 和 timestep-sensitive clipping 往往比 weight-only GPTQ 更关键。

#### AWQ

AWQ 通过 activation-aware 的方式识别对模型输出更重要的 weight channel，并保留或缩放少量 salient weights，从而实现硬件友好的 LLM 低比特 weight-only quantization [R9]。

对 Video DiT 的启示：

- 可以借鉴 activation-aware saliency；
- 但 Video DiT 的 saliency 不应只来自平均 activation magnitude，还应考虑 timestep、token motion saliency 和最终视频质量敏感性。

#### SmoothQuant

SmoothQuant 是 training-free PTQ 方法，通过数学等价的 weight/activation rescaling，将 activation outlier 平滑迁移到 weight，从而实现 LLM W8A8 量化 [R10]。

对 Video DiT 的启示：

- 可用于缓解 activation outlier；
- 但 W4A4 下单纯 smoothing 往往不足，因为迁移到 weight 后的 outlier 会加重 W4 weight quantization error；
- SVDQuant 正是针对这个问题提出 low-rank branch 吸收迁移后的 weight outlier [R11]。

#### QuaRot / rotation-based quantization

QuaRot 通过正交旋转消除 LLM hidden state outlier，使权重、activation 和 KV cache 都更容易 4bit 量化 [R12]。后续 DiT 方向也出现了 rotation-aware 方法，如 DiRotQ、ConvRot 等 [R13][R14]。

对 Video DiT 的启示：

- Hadamard / orthogonal rotation 可降低 activation peak；
- 但静态 rotation 未必能适配不同 timestep 的激活主方向；
- 对视频模型，rotation 还需考虑 row-wise outlier、channel-wise outlier 和 motion-sensitive token。

---

### 2.2 Diffusion / DiT 量化方法

#### PTQ4DM

PTQ4DM 较早指出 diffusion 模型输出分布随 timestep 改变，普通 PTQ 方法难以直接适配 diffusion 的多 timestep 结构 [R15]。

启示：timestep variance 是 diffusion 量化的基础难点，不能用单 timestep / 单分布假设做 calibration。

#### PTQ4DiT

PTQ4DiT 专门面向 Diffusion Transformer，发现 DiT linear 层存在 salient channels，并且 activation salient channels 随 timestep 明显变化。它提出 Channel-wise Salience Balancing 和 Spearman’s ρ-guided Salience Calibration，实现 W8A8 和 W4A8 DiT 量化 [R3]。

启示：

- channel salience 是 DiT 量化的重要变量；
- timestep-aware calibration 是必要的；
- 但 PTQ4DiT 仍偏向 channel/range 层面的校准，尚未显式建模量化误差对后续 denoising trajectory 的放大。

#### Q-DiT

Q-DiT 指出 DiT 存在 weight/activation 的 spatial variance，以及 activation 的 temporal variance。它提出 automatic quantization granularity allocation 和 sample-wise dynamic activation quantization [R4]。

启示：

- activation quantization 应该 sample-wise / timestep-wise 动态调整；
- granularity allocation 不应固定为 per-tensor 或 per-channel；
- 可进一步扩展为 layer × timestep × token 的敏感性分配。

#### ViDiT-Q

ViDiT-Q 是面向 video/image DiT 的量化方法，分析了 token-wise、CFG-wise、timestep-wise、input-channel-wise variance，并提出 token-wise quantization、dynamic activation quantization、timestep-aware channel balancing 和 metric-decoupled mixed precision [R5][R16]。论文还指出，在结合 FlashAttention 后，DiT 中 linear layers 是主要 latency 和 memory cost 来源，因此量化 linear / MLP 层收益显著 [R5]。

启示：

- Video DiT 必须考虑 token-level variance；
- layer type 对 visual quality、text-video alignment、temporal consistency 的影响不同；
- mixed precision 的分配目标不应只看 MSE，而应结合生成指标。

#### SVDQuant / Nunchaku

SVDQuant 面向 4-bit diffusion models，将 activation outlier 迁移到 weight 后，用 high-precision low-rank branch 吸收 weight outlier，主分支做 4bit 量化 [R11]。Nunchaku 通过 kernel fusion 减少 low-rank branch 带来的额外访存开销 [R11][R17]。

启示：

- low-rank compensation 是 W4A4 diffusion 量化中非常强的 baseline；
- 但原始 SVDQuant 的 low-rank branch 多为 layer-level static compensation；
- 对 Video DiT，low-rank residual 的方向和 rank budget 应当与 timestep / denoising phase 相关。

---

### 2.3 Video Diffusion Transformer 量化方法

#### DVD-Quant

DVD-Quant 是 data-free Video DiT quantization 方法，提出 Bounded-init Grid Refinement、Auto-scaling Rotated Quantization 和 δ-Guided Bit Switching，目标是在不依赖 heavy calibration data 的情况下实现 Video DiT W4A4 PTQ [R6]。

启示：

- data-free / low-calibration 是实际部署友好的方向；
- rotation 与 adaptive bit-width allocation 对 Video DiT 有效；
- 但 data-free 方法对 bad case coverage 的解释能力可能不足。

#### Q-VDiT

Q-VDiT 指出 image diffusion quantization 方法不能很好泛化到 video generation，因为 video 任务存在信息损失和优化目标错配问题。它提出 Token-aware Quantization Estimator 和 Temporal Maintenance Distillation，以保护跨帧时空相关性 [R18]。

启示：

- video quantization 的目标函数不能只看单帧质量；
- temporal consistency / scene consistency 应进入训练或校准目标；
- token-aware compensation 是视频 DiT 的核心方向。

#### S²Q-VDiT

S²Q-VDiT 观察到视频 diffusion 的长 token 序列会导致 high calibration variance 和 learning challenges。它提出 Hessian-aware Salient Data Selection 和 Attention-guided Sparse Token Distillation，在 W4A6 下实现 lossless performance、3.9× model compression 和 1.3× inference acceleration [R7]。

启示：

- calibration data 的选择会显著影响 Video DiT 量化结果；
- salient token / sparse token distillation 可以降低长视频 token 序列带来的校准方差；
- bad-case mining 是一个值得进一步强化的方向。

#### Wan2.2 相关 W4A4 量化工作

近期已有若干直接面向 Wan2.2 的 W4A4 量化探索：

- **Timestep-Aware SVDQuant-GPTQ**：针对 Wan2.2-I2V，结合 SVDQuant low-rank compensation、GPTQ residual weight quantization 和 timestep-bin-wise activation clipping search，并指出 Wan2.2 中 sparse activation outlier 与 timestep-dependent activation distribution 是 W4A4 的主要难点 [R19]。
- **W4A4 Quantization for Wan2.2-I2V-A14B**：结合 SmoothQuant-style smoothing、MixQ-style outlier branch 和 block-wise HiF4 packing，重点处理 FFN linear layers [R20]。
- **Tail-Aware HiFloat4**：基于 ViDiT-Q pipeline 适配 Wan2.2，引入 activation-tail-aware percentile calibration [R21]。

这些工作说明，SVDQuant + GPTQ + timestep clipping 已经成为非常强的直接 baseline。如果继续沿着这条路线做小修小补，创新性会不足。更有潜力的方向是：**从局部重构误差转向 denoising trajectory error amplification。**

---

## 3. 研究假设

### 3.1 现象

在 Wan2.1-14B MLP 线性层上使用 W4A4 / W4A8 时，即使采用 SVDQuant、Hadamard rotation、outlier clipping、SmoothQuant-style scaling 等方法，仍会出现 bad case，例如：

- 运动区域 flicker；
- 主体结构崩坏；
- 局部纹理异常；
- prompt alignment 下降；
- late timestep 细节 refinement 失败；
- 某些 seed/prompt 下明显劣化，而平均指标不敏感。

### 3.2 核心假设

**Video DiT 量化 bad case 的根因不是单层局部 MSE 最大，而是某些 layer × timestep × token 的量化误差具有更高 trajectory amplification factor。**

也就是说，某层 MLP 的局部输出误差：

\[
\Delta y_{l,t}=X_{l,t}W_l - Q(X_{l,t})Q(W_l)
\]

是否危险，不仅取决于 \(\|\Delta y_{l,t}\|\)，还取决于它在后续 denoising steps 中被放大的程度：

\[
z_{t-1}=F_\theta(z_t,t,c), \quad
z_0=F_\theta^{t\rightarrow 0}(z_t,t,c)
\]

因此，真正应该优化的是：

\[
\min_Q \sum_{l,t} \alpha_{l,t}\cdot
\left\|X_{l,t}W_l - Q(X_{l,t})Q(W_l)\right\|_2^2
\]

其中 \(\alpha_{l,t}\) 表示 layer \(l\)、timestep \(t\) 的量化误差对最终生成轨迹的放大系数。

---

## 4. 推荐主算法：Trajectory-Aware Low-Bit Quantization for Video DiT

建议将论文主线命名为：

**TACQ: Trajectory-Aware Compensation Quantization for Low-Bit Video Diffusion Transformers**

或：

**TrajQuant: Trajectory-Aware Low-Bit Quantization for Video Diffusion Transformers**

整体框架包含四个模块：

1. **Trajectory Amplification Estimator**：估计 layer × timestep 的误差放大系数；
2. **Timestep-Conditioned Low-Rank Error Bank**：替代 static SVDQuant low-rank branch；
3. **Functional MLP Joint Quantization**：对 gate/up/down FFN 整体量化，而不是逐线性层独立量化；
4. **Bad-case Active Calibration**：用生成失败模式指导 calibration set 和 timestep selection。

---

## 5. 模块一：Trajectory Amplification Estimator

### 5.1 动机

现有 GPTQ、AWQ、SVDQuant、rotation 等方法多优化局部误差：

\[
\min \|XW - \hat{X}\hat{W}\|_2^2
\]

但 Video DiT 中某些 timestep 的小误差可能被后续 denoising 过程放大，导致最终视频 bad case。因此需要估计：

\[
\alpha_{l,b}
= \frac{\text{final video / latent degradation}}{\text{injected local error magnitude}}
\]

其中 \(b\) 是 timestep bin。

### 5.2 估计方式

可选三种实现，由轻到重：

#### 方法 A：Current-step output sensitivity

在 FP16 teacher trajectory 上缓存 layer input/output，对 MLP output 注入小扰动：

\[
h_l' = h_l + \epsilon
\]

只测当前 block 或当前 timestep model output 的变化：

\[
\alpha_{l,b} = \mathbb{E}_{t\in b}\left[
\frac{\|F_\theta^{q}(z_t,t,c;h_l+\epsilon)-F_\theta(z_t,t,c)\|_2}
{\|\epsilon\|_2}
\right]
\]

优点：成本低；缺点：没有完整多 step propagation。

#### 方法 B：Short-horizon denoising sensitivity

从 timestep \(t\) 开始，扰动某层 MLP output，然后继续运行 \(k\) 个 denoising steps：

\[
\alpha_{l,b}^{(k)} =
\frac{\|z_{t-k}'-z_{t-k}\|_2}{\|\epsilon\|_2}
\]

优点：能近似误差传播；缺点：成本中等。

#### 方法 C：Bad-case metric proxy

对完整生成视频计算 VBench 子指标或轻量 proxy：

- temporal flicker proxy；
- subject consistency proxy；
- frame-wise CLIP similarity；
- latent trajectory drift；
- optical-flow consistency proxy。

然后根据量化/扰动前后的质量差得到 \(\alpha_{l,b}\)。

优点：最接近最终目标；缺点：成本最高。

### 5.3 预期观察

关键实验应该证明：

1. 高 amplification layer 不一定是 activation max 最大的层；
2. 高 amplification timestep 不一定是 local MSE 最大的 timestep；
3. 用 amplification-weighted calibration 比 MSE-weighted calibration 更能减少 bad case；
4. MLP down projection、late timestep texture refinement、motion-salient tokens 可能具有更高 amplification。

---

## 6. 模块二：Timestep-Conditioned Low-Rank Error Bank

### 6.1 动机

SVDQuant 使用静态 low-rank branch：

\[
W \approx Q_4(W_{res}) + UV^T
\]

但 Video DiT activation distribution 强依赖 timestep，不同 denoising phase 的 outlier 方向和误差方向可能不同。静态 low-rank residual 难以同时覆盖 early/mid/late timesteps。

### 6.2 方法

提出 timestep-conditioned low-rank error bank：

\[
W \approx Q_4(W_{res}) + \sum_{k=1}^{K} g_k(t) U_kV_k^T
\]

其中：

- \(K\)：timestep group / denoising phase 数量；
- \(g_k(t)\)：由 timestep bin lookup 或 time embedding MLP 产生的 gate；
- \(U_kV_k^T\)：第 \(k\) 个 timestep group 的 low-rank compensation。

### 6.3 Training-free 版本

1. 收集 calibration prompts / seeds / timesteps；
2. 将 timesteps 聚类为 early / mid / late，或使用 hierarchical timestep grouping；
3. 对每个 group \(b\) 缓存 MLP input \(X_b\)；
4. 先得到主分支 W4 quantized weight \(Q_4(W)\)；
5. 计算 group-specific residual：

\[
E_b = X_b(W-Q_4(W))
\]

6. 对 residual 做 SVD 或 covariance-weighted SVD；
7. 根据 trajectory amplification factor 分配 rank budget：

\[
r_{l,b} \propto \alpha_{l,b}\cdot \text{ResidualEnergy}_{l,b}
\]

### 6.4 Training-aware 版本

只训练轻量参数：

- low-rank branch \(U_k,V_k\)；
- timestep gate \(g_k(t)\)；
- scale / zero point / clip ratio；
- optional learned rotation。

保持主模型 weight frozen。

损失函数：

\[
\mathcal{L}=
\lambda_1\|\epsilon_\theta^q(z_t,t,c)-\epsilon_\theta^{fp}(z_t,t,c)\|_2^2
+\lambda_2\sum_l\|h_l^q-h_l^{fp}\|_2^2
+\lambda_3\mathcal{L}_{temporal}
\]

其中 \(\mathcal{L}_{temporal}\) 可使用相邻帧 latent consistency、optical flow consistency 或 attention map consistency。

### 6.5 创新点

相对 SVDQuant：

- SVDQuant 是 layer-level static low-rank compensation；
- 本方法是 timestep-conditioned low-rank error bank；
- rank allocation 由 denoising trajectory sensitivity 决定，而非仅由 weight residual energy 决定。

---

## 7. 模块三：Functional MLP Joint Quantization

### 7.1 动机

Wan 类 DiT 的 FFN/MLP 通常可抽象为：

\[
\text{FFN}(x)=W_d\left(\text{SiLU}(xW_g)\odot xW_u\right)
\]

如果逐层独立量化 \(W_g,W_u,W_d\)，会忽略 gate/up/down 之间的误差相关性。尤其是 SiLU 和 elementwise product 会非线性放大 gate/up projection 的误差。

### 7.2 方法

将整个 FFN 作为函数单元进行量化：

\[
\min_{Q(W_g),Q(W_u),Q(W_d)}
\sum_{t}\left\|
\text{FFN}_{fp}(x_t)-\text{FFN}_{q}(x_t)
\right\|_2^2
\]

进一步加入 trajectory amplification：

\[
\min
\sum_{b}\alpha_{l,b}
\left\|
\text{FFN}_{fp}(x_{l,b})-
\text{FFN}_{q}(x_{l,b})
\right\|_2^2
\]

### 7.3 近似求解

1. 先量化 gate/up projection，得到中间激活误差：

\[
\Delta h = \text{SiLU}(xQ(W_g))\odot xQ(W_u)
- \text{SiLU}(xW_g)\odot xW_u
\]

2. 量化 down projection 时，不拟合 \(hW_d\)，而拟合：

\[
h_q Q(W_d) \approx h W_d
\]

使 down projection 的 rounding error 部分抵消 gate/up 误差。

3. 对 SiLU 高导数区和饱和区采用不同 activation clipping policy：

- 高导数区：保留更多精度；
- 饱和区：允许更 aggressive clipping；
- motion-salient token：使用更宽 clip range 或 A6/A8 fallback。

### 7.4 创新点

- 从 layer-wise reconstruction 转向 function-wise MLP reconstruction；
- 显式建模 gate/up/down 的误差补偿；
- 更适合 DiT MLP，而不是简单把所有 Linear 当作独立 GEMM。

---

## 8. 模块四：Bad-case Active Calibration

### 8.1 动机

S²Q-VDiT 已经指出 Video Diffusion Model 对 calibration data 非常敏感，长 token 序列会带来高 calibration variance [R7]。因此，随机 calibration prompts 可能无法覆盖真实 bad case。

### 8.2 方法

构造闭环：

\[
\text{Quantize} \rightarrow \text{Generate} \rightarrow \text{Mine bad cases} \rightarrow \text{Recalibrate}
\]

步骤：

1. 使用当前 W4A4 / W4A8 模型生成一批视频；
2. 与 BF16 teacher 对比，自动检测 bad cases；
3. 对 bad cases 回溯 layer × timestep × token error map；
4. 将 bad cases 对应 prompts、seeds、timesteps 加入 calibration set；
5. 重新优化 clip ratio、scale、rotation、low-rank rank allocation。

### 8.3 Bad-case 检测指标

可使用：

- frame-wise latent drift；
- temporal flicker score；
- CLIP text-video alignment drop；
- frame-to-frame feature consistency；
- VBench 子指标下降；
- optical flow inconsistency；
- motion-salient region error。

### 8.4 创新点

- calibration 目标从“覆盖 activation range”升级为“覆盖生成失败模式”；
- 对 bad case rate 的改善可能比平均 VBench 更明显；
- 适合解释你当前实验中“平均还行但总有坏样本”的现象。

---

## 9. 可选增强方向：Spatio-Temporal Salient Token Activation Quantization

### 9.1 动机

Video DiT token 具有空间和时间结构，不同 token 对最终视频质量的贡献不一样。运动区域、主体边界、cross-attention salient token 量化错误更容易造成主观质量下降。

### 9.2 方法

将 video latent tokens 分为：

1. motion-salient tokens：相邻帧 latent 差异大；
2. attention-salient tokens：self/cross-attention centrality 高；
3. foreground/object tokens：局部激活能量高、attention entropy 低；
4. background/static tokens：可 aggressive quantization。

量化策略：

\[
Y = Q_4(X_{normal})Q_4(W) + Q_8(X_{salient})Q_4(W_{salient})
\]

或者：

- normal tokens: W4A4；
- motion/foreground salient tokens: W4A6 / W4A8；
- extremely sensitive layers/timesteps: sparse FP16 residual branch。

### 9.3 与现有方法区别

ViDiT-Q 已经有 token-wise quantization，S²Q-VDiT 有 attention-guided sparse token distillation [R5][R7]。本方向的差异化在于：

- saliency 不只来自 attention，还来自 motion / temporal inconsistency；
- saliency 直接服务于 activation precision allocation；
- 与 trajectory amplification factor 结合，决定 token-level fallback 策略。

---

## 10. 建议论文结构

### 标题候选

1. **TrajQuant: Trajectory-Aware Low-Bit Quantization for Video Diffusion Transformers**
2. **TACQ: Trajectory-Aware Compensation Quantization for W4A4 Video DiT Inference**
3. **T-LoRAQ: Timestep-Conditioned Low-Rank Error Compensation for Low-Bit Video Diffusion Transformers**
4. **From Local Reconstruction to Trajectory Preservation: Low-Bit Quantization for Video DiTs**

### Abstract 主线

现有 Video DiT PTQ 方法主要优化局部 layer reconstruction error 或 activation range calibration，但 W4A4/W4A8 量化失败往往来自多 timestep denoising trajectory 中的误差放大。本文提出 trajectory-aware quantization framework，显式估计 layer × timestep 的 error amplification factor，并据此指导 timestep-conditioned low-rank compensation、functional MLP joint quantization 和 bad-case active calibration。在 Wan2.1/Wan2.2、Open-Sora 或 HunyuanVideo 上验证，该方法可在维持低 bit inference efficiency 的同时降低 bad-case rate，并提升 temporal consistency 与 subject consistency。

### 主要贡献写法

1. We identify trajectory amplification as a key source of bad cases in low-bit Video DiT quantization.
2. We propose a Jacobian-free trajectory amplification estimator to measure layer- and timestep-wise quantization sensitivity.
3. We introduce timestep-conditioned low-rank error banks to compensate W4 residual errors across different denoising phases.
4. We formulate MLP quantization as functional FFN reconstruction instead of independent linear-layer reconstruction.
5. We design bad-case active calibration to reduce quantization-induced video generation failures.

---

## 11. 实验设计

### 11.1 模型

建议分三阶段：

1. **快速验证**：Wan2.1-1.3B 或较小 Video DiT；
2. **主实验**：Wan2.1-14B，重点量化 MLP/FFN linear；
3. **泛化实验**：Wan2.2、HunyuanVideo、Open-Sora / Latte / CogVideoX 等。

### 11.2 量化设置

| 设置 | 说明 |
|---|---|
| BF16 | teacher baseline |
| W8A8 | sanity check |
| W4A8 | 主要稳定目标 |
| W4A4 | 主要挑战目标 |
| W4A4 + FP16 sensitive boundary | 工程可落地方案 |
| W4A4 + sparse salient token A8 fallback | 质量优先方案 |

### 11.3 Baselines

建议至少包含：

- RTN / MinMax；
- SmoothQuant-style smoothing；
- GPTQ；
- AWQ-style activation-aware scaling；
- Hadamard / QuaRot-style rotation；
- SVDQuant；
- ViDiT-Q；
- PTQ4DiT / Q-DiT；
- DVD-Quant；
- S²Q-VDiT；
- Wan2.2 SVDQuant-GPTQ / HiF4 challenge baselines，如果代码可用。

### 11.4 指标

建议报告：

- VBench overall score；
- Imaging Quality；
- Aesthetic Quality；
- Motion Smoothness；
- Dynamic Degree；
- Subject Consistency；
- Background Consistency；
- Scene Consistency；
- text-video alignment；
- bad-case rate；
- memory footprint；
- end-to-end latency；
- MLP GEMM latency；
- kernel overhead of low-rank branch / token fallback。

### 11.5 Ablation

必须做的 ablation：

1. local MSE sensitivity vs trajectory amplification sensitivity；
2. static SVDQuant vs timestep-conditioned low-rank bank；
3. layer-wise GPTQ vs functional MLP joint quantization；
4. random calibration vs bad-case active calibration；
5. early/mid/late timestep bins；
6. rank budget uniform vs amplification-weighted；
7. gate/up/down 分别量化敏感性；
8. motion-salient token fallback 是否有效；
9. W4A8 与 W4A4 下方法收益差异。

---

## 12. 最小可行实验 MVP

为了尽快验证论文核心假设，建议先做以下 MVP：

1. 选择 Wan2.1-1.3B 或 Wan2.1-14B 的 5–10 个 MLP down projection 敏感层；
2. 在 FP16 denoising trajectory 上缓存每个 timestep bin 的 MLP activation；
3. 对每个 layer × timestep bin 注入等范数 Gaussian perturbation；
4. 测最终 latent drift、短 horizon drift 或 VBench proxy；
5. 画出 layer × timestep amplification heatmap；
6. 证明 amplification heatmap 与 local MSE / activation max / outlier ratio 不完全一致；
7. 用 amplification score 指导 rank allocation 或 clip search；
8. 对比 bad-case rate 是否下降。

如果第 6 点和第 8 点成立，论文故事基本成立。

---

## 13. 风险与应对

### 风险 1：trajectory amplification 估计成本高

应对：先用 short-horizon denoising 或 current-step output sensitivity；只对 MLP sensitive layers 估计，而不是全模型全层。

### 风险 2：timestep-conditioned low-rank branch 增加 runtime overhead

应对：

- 只在少数 sensitive layers 使用；
- 使用 low rank，例如 r=4/8/16；
- 与主 GEMM kernel fusion；
- 或离线 fold 到 grouped expert-like branch。

### 风险 3：token-level mixed precision 不硬件友好

应对：

- 先做 fake quant 证明质量收益；
- 再改成 block-token fallback，例如固定 block size 的 salient tile；
- 或只在 activation quantization scale 上做 token group，不真的动态分支。

### 风险 4：创新性被 Wan2.2 SVDQuant-GPTQ 覆盖

应对：明确差异：

- 他们做 timestep-bin clip search + SVDQuant + GPTQ；
- 你的核心是 trajectory amplification estimator + trajectory-weighted compensation；
- 不是只根据 activation range / clipping ratio 做校准，而是根据最终 denoising 误差传播做优化。

---

## 14. 推荐下一步实现路线

### Step 1：复现实验现象

- 固定 prompts/seeds；
- 只量化 MLP linear；
- 保存 FP16 与 W4A4/W4A8 的 latent trajectory；
- 分类 bad case。

### Step 2：构造 amplification heatmap

- layer × timestep perturbation；
- 输出 short-horizon drift / final latent drift；
- 与 local MSE、activation max、outlier ratio 对比。

### Step 3：做 amplification-aware rank allocation

- 在 SVDQuant 或 GPTQ residual 上加权；
- 只改 rank allocation 和 clip ratio；
- 快速验证 bad case rate。

### Step 4：实现 timestep-conditioned low-rank bank

- 先用 discrete timestep bin lookup；
- 后续再换 time embedding gate；
- 比较 static low-rank vs timestep-conditioned low-rank。

### Step 5：写论文核心实验

- 主模型 Wan2.1-14B；
- W4A8 + W4A4；
- 对比 ViDiT-Q、SVDQuant、DVD-Quant、S²Q-VDiT；
- 强调 bad-case rate、temporal consistency 和 subject consistency。

---

## 参考文献

[R1] Wan-Video/Wan2.1 GitHub repository. https://github.com/Wan-Video/Wan2.1

[R2] OpenLaboratory, “Wan 2.1 I2V 14B 720P,” architecture summary. https://openlaboratory.com/models/wan2_1-i2v-14b-720p/

[R3] Junyi Wu, Haoxuan Wang, Yuzhang Shang, Mubarak Shah, Yan Yan. “PTQ4DiT: Post-training Quantization for Diffusion Transformers.” arXiv:2405.16005, 2024. https://arxiv.org/abs/2405.16005

[R4] Lei Chen, Yuan Meng, Chen Tang, Xinzhu Ma, Jingyan Jiang, Xin Wang, Zhi Wang, Wenwu Zhu. “Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers.” arXiv:2406.17343, 2024. https://arxiv.org/abs/2406.17343

[R5] Tianchen Zhao et al. “ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation.” arXiv:2406.02540, 2024. https://arxiv.org/abs/2406.02540

[R6] Zhiteng Li et al. “DVD-Quant: Data-free Video Diffusion Transformers Quantization.” arXiv:2505.18663, 2025. https://arxiv.org/abs/2505.18663

[R7] Weilun Feng et al. “S²Q-VDiT: Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation.” arXiv:2508.04016, 2025. https://arxiv.org/abs/2508.04016

[R8] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh. “GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.” arXiv:2210.17323, 2022. https://arxiv.org/abs/2210.17323

[R9] Ji Lin et al. “AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.” arXiv:2306.00978, 2023. https://arxiv.org/abs/2306.00978

[R10] Guangxuan Xiao et al. “SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.” arXiv:2211.10438, 2022. https://arxiv.org/abs/2211.10438

[R11] Muyang Li et al. “SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models.” arXiv:2411.05007, 2024. https://arxiv.org/abs/2411.05007

[R12] Saleh Ashkboos et al. “QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.” arXiv:2404.00456, 2024. https://arxiv.org/abs/2404.00456

[R13] Sayeh Sharify et al. “Rotation-Aware Quantization for 4-bit Diffusion Transformers.” arXiv:2605.16732, 2026. https://arxiv.org/abs/2605.16732

[R14] Feice Huang et al. “ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers.” arXiv:2512.03673, 2025. https://arxiv.org/abs/2512.03673

[R15] Yuzhang Shang, Zhihang Yuan, Bin Xie, Bingzhe Wu, Yan Yan. “Post-training Quantization on Diffusion Models.” arXiv:2211.15736, 2022. https://arxiv.org/abs/2211.15736

[R16] ViDiT-Q GitHub repository. https://github.com/thu-nics/ViDiT-Q

[R17] Nunchaku GitHub repository. https://github.com/nunchaku-ai/nunchaku

[R18] Weilun Feng et al. “Q-VDiT: Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers.” arXiv:2505.22167, 2025. https://arxiv.org/abs/2505.22167

[R19] Junhao Wu, Dezhong Yao, Hai Jin. “Timestep-Aware SVDQuant-GPTQ for W4A4 Quantization of Wan2.2-I2V.” arXiv:2605.27003, 2026. https://arxiv.org/abs/2605.27003

[R20] Yidong Chen, Chengyu Shi, Jiahao Liu. “W4A4 Quantization for Inference on Wan2.2-I2V-A14B.” arXiv:2606.29337, 2026. https://arxiv.org/abs/2606.29337

[R21] Zhanfeng Feng, Shuai Guo, Xin Di, Long Peng, Yang Cao, Zhengjun Zha. “Tail-Aware HiFloat4: W4A4 Post-Training Quantization for Wan2.2.” arXiv:2605.26628, 2026. https://arxiv.org/abs/2605.26628
