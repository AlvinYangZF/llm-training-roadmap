# Coupled Quantization 论文深度解读

**论文:** KV Cache is 1 Bit Per Channel: Efficient Large Language Model Inference with Coupled Quantization
**作者:** Tianyi Zhang, Jonah Yi, Zhaozhuo Xu, Anshumali Shrivastava
**机构:** Rice University, Stevens Institute of Technology, ThirdAI Corp.
**arXiv:** 2405.03917
**发表时间:** 2024年5月

---

## 一、解决什么问题？

```
现有KV Cache量化的根本问题：通道独立量化忽视了通道间的依赖性

现有方法的量化策略:
  Per-channel量化 (KIVI, KVQuant):
    每个通道独立学习量化centroids
    → 假设各通道相互独立
    → 1-2bit时精度急剧崩溃

  Per-token量化:
    每个token独立量化其所有通道
    → 同样假设独立性
    → 极低精度下同样失效

为什么独立量化在极低精度失效？
  → 1bit per channel = 每个通道只有2个量化级别！
  → 2个点无法描述连续分布 → 信息损失巨大

关键洞察（信息论视角）:
  H(X_1, X_2) ≤ H(X_1) + H(X_2)
  （联合熵 ≤ 边缘熵之和）

  如果 X_1 和 X_2 高度相关：
  H(X_1, X_2) << H(X_1) + H(X_2)

  → 联合编码多个通道比独立编码更信息高效！
  → 相关通道联合量化 = 用更少的比特表示相同信息
```

---

## 二、核心方法/关键发现

### 关键发现1：KV Cache通道间具有高度依赖性

```
实验验证 (LLaMA-7b, WikiText-2, 262k tokens):

  联合熵 vs 边缘熵之和 (不同层):

         Layer 1          Layer 2          Layer 3
  Key  ▲                  ▲                ▲
  熵   │  橙线(边缘熵和)    │                │
 (bits) │  ╱              │  ╱              │  ╱
       │ ╱  蓝线(联合熵)   │ ╱               │ ╱
       │╱                 │╱                │╱
       └──────→           └──────→          └──────→
         通道数量             通道数量           通道数量

  关键观察:
    → 联合熵增长速率 << 边缘熵之和增长速率
    → 差距随通道数增加而扩大
    → Key和Value均呈现此规律

  定量结论: 联合量化4个通道比独立量化4个通道
    需要的信息量更少 → 相同比特数可获得更好精度
```

### 关键发现2：通道间存在强线性依赖

```
Pearson相关系数矩阵 (LLaMA-7b前32个通道, 多个层):

  Layer 1 Key:        Layer 13 Key:       Layer 29 Key:
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │██████████████│   │████████      │   │██████        │
  │██████████████│   │████████      │   │██████        │
  │██████████████│   │    ████      │   │      ████    │
  │██████████████│   │    ████      │   │      ████    │
  └──────────────┘   └──────────────┘   └──────────────┘
  颜色: 深红=高正相关, 深蓝=高负相关

  → 浅层: 大块通道块高度相关（接近全部正相关）
  → 深层: 多个独立通道组，组内高度相关
  → 通道间明显不独立！现有方法假设独立是错误的
```

### CQ 方法的核心思想

```
Coupled Quantization (CQ) 的直觉图解:

  Per-Channel量化 (1-bit, 每通道独立):
  Channel 0: centroid = [0.21]      错误率: 601.4
  Channel 1: centroid = [-0.38]
  → 每个通道只有1个中心点
  → 点落在哪里就量化到最近的那个 (2个级别)
  → 无法捕捉2D联合分布的形状

  Coupled Quantization (2bit per 2 channels):
  Channels {0,1}: centroid = [(0.11,0.30), (-0.38,-0.10), ...]
  → 2通道联合 → 2D空间
  → 8个中心点（2bit×2通道 = 4bit总代码）
  → 中心点分布沿相关轴方向 → 捕捉联合分布
  → 错误率: 250.6  ← 降低 58%！

CQ表示法:  CQ-<c><b>b
  <c>: 每组耦合的通道数
  <b>: 每组的量化代码位数
  等效bit宽: b/c bits per channel

  CQ-4c8b = 每4通道8位码 = 2 bits/channel (存储等效)
  CQ-8c8b = 每8通道8位码 = 1 bit/channel
  CQ-8c10b = 每8通道10位码 = 1.25 bits/channel
```

---

## 三、技术细节

### Centroid 学习

**均匀Centroid学习（Uniform CQ）：**

```
对每个通道组 i，学习 c 维空间中的 2^b 个中心点：

  C_i* = argmin_{|C|=2^b} ||A_{ic:(ic+c-1),*} - cq(A_{ic:(ic+c-1),*})||_F²

其中 cq() 将每个列向量映射到最近的中心点

优化方法: k-means (k-means++ 初始化)
```

**Fisher引导的Centroid学习（Fisher-guided CQ）：**

```
LLM对不同激活的量化精度敏感程度不同
→ 重要的激活应该被更精确地量化

Fisher信息矩阵近似:
  F = diag(g(A) ⊙ g(A))   (梯度的平方作为重要性权重)
  其中 g(A) = ∂L/∂A

加权k-means目标:
  C_i* = argmin_{|C|=2^b} Σ_j g(A_{ic:(ic+c-1),j})^T × g(A_{ic:(ic+c-1),j})
          × ||A_{ic:(ic+c-1),j} - cq(A_{ic:(ic+c-1),j})||_F²

→ 重要激活（大梯度）有更大的量化权重
→ 中心点向重要数据点聚集
→ 更好地保持模型质量（即使量化误差增大）
```

### 推理过程

```
预处理阶段 (每个模型部署前):
  1. 从校准集采样 (16个序列, 每个2048 tokens, WikiText-2)
  2. 对每层每个通道组，学习2^b个多维中心点
  3. 存储 centroid 查找表 (存储开销: 每层 num_groups × 2^b × c 个fp16值)

推理时量化:
  新KV向量 a 到达:
  1. 按通道分组: (a_0, a_1, ..., a_{c-1}), (a_c, ...) ...
  2. 对每组，找最近的 centroid (L2距离): idx = argmin_j ||a_{group} - C_j||
  3. 存储 idx (b bits)

推理时反量化:
  取出idx → 查表得 centroid → 作为反量化结果
  计算注意力: q^T * dequant(K), softmax(...) * dequant(V)

额外存储开销:
  centroids大小 << KV Cache大小
  → 例如: 2048通道 × 512组 × 2^8 × 4 channels × fp16
         = 约 16MB per layer  (vs KV Cache可能数GB)
```

### 关键设计选择

```
1. 相邻通道耦合（非随机）:
   → 论文发现相邻通道更可能高度相关
   → 按顺序分组 {0,1,...,c-1}, {c,...,2c-1}, ...
   → 简单且有效

2. Keys在RoPE前量化（与KVQuant一致）:
   → Pre-RoPE Keys有更整洁的通道结构
   → RoPE破坏相邻通道的相关性
   → Pre-RoPE量化让CQ更有效

3. Keys和Values都使用Per-Channel耦合（而非Value per-token）:
   → 与KIVI不同，CQ对K和V都使用通道耦合
   → 通道耦合对Value同样有效
   → 统一的量化框架，更简洁
```

---

## 四、实验结果

### 困惑度 (PPL) 主要结果

```
WikiText-2 测试结果 (FP16 baseline对照):

  模型: LLaMA-7b (FP16: 5.68)

  位宽   方法              PPL
  4bit  INT4              5.98
        KVQuant-4b        5.73
        KVQuant-4b-1%     5.70
        CQ-2c8b (=4bit)   5.70   ← 与KVQuant-4b-1%持平！

  2bit  INT2              11779 (崩溃)
        KVQuant-2b        8.17
        KVQuant-2b-1%     6.06
        CQ-4c8b (=2bit)   5.97   ← 优于KVQuant-2b-1%！

  1bit  KVQuant-1b        321.58 (几乎崩溃)
        KVQuant-1b-1%     9.93
        CQ-8c8b (=1bit)   8.09   ← 大幅优于KVQuant-1b-1%！
        CQ-8c10b (=1.25bit) 6.78 ← 接近2bit方法！

  LLaMA-2-7b (FP16: 5.12):
  1bit  KVQuant-1b       NaN (数值不稳定)
        KVQuant-1b-1%     9.50
        CQ-8c8b           7.75   ← KVQuant崩溃时CQ仍可用！
```

```
C4 测试结果 (LLaMA-7b, FP16: 7.08):

  2bit  KVQuant-2b        10.28
        KVQuant-2b-1%      7.38
        CQ-4c8b            7.52   ← 接近KVQuant-2b-1%

  1bit  KVQuant-1b        168.90
        KVQuant-1b-1%      11.18
        CQ-8c8b            12.13  ← 优于KVQuant-1b-1%
        CQ-8c10b            9.12  ← 远优于KVQuant-1b-1%
```

### 分类基准测试 (WinoGrande, PIQA, ARC Challenge)

```
LLaMA-7b, WinoGrande:
  FP16:           69.93
  KVQuant-4b:     69.53
  CQ-2c8b (4bit): 70.40    ← 甚至略超FP16！
  KVQuant-2b:     53.59
  KVQuant-2b-1%:  68.03
  CQ-4c8b (2bit): 67.48    ← 与KVQuant-2b-1%相当
  KVQuant-1b:     50.51
  KVQuant-1b-1%:  56.67
  CQ-8c8b (1bit): 66.51    ← 大幅优于KVQuant-1b-1%！

结论：CQ在1-bit极低精度下的优势最为突出
```

### 消融实验

```
CQ核心组件的贡献 (Mistral-7b, WikiText-2, 2bit):

  组件               PPL     改善
  独立per-channel   5.77     -
  + 通道耦合(2c→4c) 5.32   -0.45  ← 耦合的主要贡献
  + Fisher引导      5.11   -0.21  ← Fisher进一步改善

  均匀 vs Fisher-guided:
    增加耦合通道数时，两者都降低PPL
    Fisher-guided始终优于均匀centroid学习
```

---

## 五、核心启示与局限

### 核心启示

```
1. 信息论为KV Cache量化提供了新视角
   → H(X₁,X₂) ≤ H(X₁)+H(X₂) 不是trivial不等式
   → 实测中联合熵比边缘熵之和小30-50%
   → 这意味着独立编码浪费了30-50%的编码能力
   → 联合量化是逻辑上正确的方向

2. 1bit per channel 是可以做到的
   → 现有方法1bit时完全崩溃（PPL > 300）
   → CQ-8c8b (1bit) 在LLaMA-7b上PPL仅8.09
   → 1bit = 16× 压缩比相对fp16！

3. Fisher信息作为量化重要性权重是通用技巧
   → 来自权重量化领域（GPTQ等）
   → 同样适用于激活量化（KV Cache）
   → 梯度平方近似Hessian对角线是高效且准确的近似

4. Dense-and-Sparse vs Channel Coupling 是不同思路的权衡
   → KVQuant: 分离离群值到稀疏矩阵 → 需要额外稀疏矩阵乘法
   → CQ: 通过耦合天然适应离群值分布 → 无额外运算开销
   → 两者在低精度下均有效，但CQ在1bit更有优势
```

### 局限性

```
1. Centroid查找表的存储开销
   → 每层需要存储 num_groups × 2^b 个c维中心点
   → c=8, b=8: 每层 (2048/8) × 256 × 8 × 2 bytes ≈ 1MB
   → 对于大模型（70B+）的多层，centroid存储不可忽视

2. 量化过程的计算开销
   → 需要对每个新token计算最近的centroid (L2距离)
   → c=8时: 每个token-head需要比较256个8维向量
   → 需要专用kernel实现才能不成为瓶颈（论文未提供CUDA实现）

3. 离线校准的泛化性
   → 在WikiText-2上校准，在其他领域可能次优
   → 领域差异大时（代码 vs 自然语言）需要重新校准
   → 校准集大小（16个序列）是否足够未做深入分析

4. 实验缺乏延迟/吞吐量测试
   → 论文只报告PPL和分类准确率
   → 没有实际推理速度、内存使用的测试
   → 与KIVI/KVQuant相比：缺少系统级评估
   → CQ的实际加速效果未经验证
```

---

## 六、三种KV Cache量化方案对比

```
方案对比矩阵:

维度              KIVI          KVQuant         CQ (本文)
理论基础          统计分析       统计+灵敏度      信息论
核心创新          非对称量化     Pre-RoPE+非均匀  通道耦合量化
校准需求          无             需要（少量）     需要（少量）
最低可用位宽      2bit           2bit             1bit
1bit精度          崩溃           有限（稀疏补救）  可用！
推理额外开销      低             中（稀疏乘法）   待评估
实现复杂度        低             高（4种技术）    中
长上下文支持      适中           1M-10M tokens    未测试
流式支持          原生支持       原生支持         原生支持
吞吐量验证        2.35~3.47×    未报告           未报告
系统集成          vLLM插件       自定义kernel     基于PyTorch
```

---

## 七、一句话总结

> **Coupled Quantization基于信息论观察到LLM的KV Cache通道间具有强依赖性（联合熵显著小于边缘熵之和），提出将相邻通道联合量化并用Fisher引导的k-means学习多维量化centroids，突破了现有方法在1bit时精度崩溃的瓶颈，在LLaMA/LLaMA-2/Mistral模型上实现了真正可用的1bit per channel KV Cache量化（PPL ≈ 8.09 vs 基线5.68），为极致压缩KV Cache开辟了基于信息论的新方向。**

---

*解读日期：2026-04-07*
