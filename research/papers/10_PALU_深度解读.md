# PALU 论文深度解读

**论文:** PALU: KV-Cache Compression with Low-Rank Projection
**作者:** Chi-Chih Chang, Wei-Cheng Lin, Chien-Yu Lin, Chong-Yan Chen, Yu-Fang Hu, Pei-Shuo Wang, Ning-Chi Huang, Luis Ceze, Mohamed S. Abdelfattah, Kai-Chiang Wu
**机构:** National Yang Ming Chiao Tung University, University of Washington, Cornell University
**arXiv:** 2407.21118
**代码:** https://github.com/shadowpa0327/Palu

---

## 一、解决什么问题？

现有 KV cache 压缩方法主要有两类：

```
量化方法（Quantization）:
  - 降低每个元素的比特宽度（如 FP16 → INT4）
  - 代表工作: KIVI, KVQuant, Atom
  - 局限: 只压缩数值精度，无法利用 hidden dimension 中的结构冗余

Token 淘汰方法（Token Eviction）:
  - 选择性丢弃不重要 token 的 KV 对
  - 代表工作: H2O, Scissorhands, SnapKV
  - 局限: 只压缩 token 数量维度，无法利用 hidden dimension 冗余

共同盲区：
  → 两类方法都忽略了 KV 张量在 hidden dimension 方向上存在大量冗余！

```

PALU 开创了第三条压缩维度：**通过低秩分解压缩 KV 张量的 hidden dimension**，与量化和 token 淘汰完全正交，可叠加使用。

---

## 二、核心方法/关键发现

### 核心思路：缓存低秩隐表示而非原始 KV

```
标准注意力机制:
  k_i = x * W_K      (x: [d], W_K: [d×d_h])
  v_i = x * W_V      (x: [d], W_V: [d×d_h])
  缓存: K, V 张量 (全尺寸)

PALU 的做法:
  离线对 W_K 做 SVD 低秩分解: W_K ≈ A_K * B_K
    A_K ∈ [d × r],  B_K ∈ [r × d_h],  r << d_h

  推理时只缓存低秩隐表示 h:
    h = x * A_K    → 维度从 d_h 压缩到 r
  
  解码时按需重建:
    K_t = h_t * B_K  (on-the-fly 重建)

  内存节省比例 ≈ r / d_h (即低秩压缩率)
```

### 矩阵融合消除重建开销

```
对 Value 的处理（无 RoPE 时）：
  正常流程:  a_i * W_o = (p_i * V_i) * W_o
  PALU:      a_i * W_o = p_i * H_v * (B_v * W_o)
  
  关键优化: B_v 与 W_o 可以离线融合为 B_v_fused = B_v * W_o
  → 解码时无需显式重建完整 V，直接用低秩 H_v 计算输出
  → 消除重建带来的额外 FLOPs

对 Key 的处理（无 RoPE 时）：
  q_i * K_i^T = x_t * W_q * (B_k^T * H_k^T)
  融合矩阵: W_q_fused = W_q * B_k^T
  → 同样消除重建开销

对 RoPE 模型（如 LLaMA）：
  RoPE 的非线性性阻止了矩阵融合
  → PALU 设计专用 Triton kernel 在线重建 Key 后再施加 RoPE
  → 仍能获得 1.89× 加速
```

---

## 三、技术细节

### 分解粒度：三种方案的权衡

```
方案A: Multi-Head LRD (M-LRD)
  对每个注意力头独立做 SVD 分解
  优点: 重建 FLOPs 最低（O(r_i * d_h)）
  缺点: 不同头独立分解，丢失头间共享信息 → 精度损失大

方案B: Joint-Head LRD (J-LRD)
  对所有头的拼接权重矩阵联合做 SVD
  优点: 保留头间共享主成分 → 精度最好
  缺点: 重建 FLOPs 是 M-LRD 的 n 倍 → 延迟高

方案C: Group-Head LRD (G-LRD) ← PALU 默认选择
  将 n 个头分成 n/s 组，每组 s 个头联合分解
  组大小 gs=4 时的折中效果：
    精度接近 J-LRD
    FLOPs 和内存只有 J-LRD 的 1/s

  G-LRD 重建 FLOPs = r_g * d_h * n_g，其中 n_g = n/s
  → 在精度与效率之间取得最优平衡
```

### 自动秩搜索（基于 Fisher 信息）

```
不同线性层对压缩的敏感度不同
  → 直接分配均匀秩会浪费重要层的精度

PALU 的解决方案：
  1. 计算每个权重矩阵的 Fisher 信息（近似重要性度量）
     Fisher(W_i) = sum of (grad^2 * input^2) over calibration data
  
  2. 总压缩率固定的情况下，按 Fisher 信息比例分配秩：
     rank_i ∝ Fisher(W_i) / sum_j(Fisher(W_j))
  
  3. 重要的层获得更高秩（更多参数保留信息）
     不重要的层获得更低秩（激进压缩）

效果: 在相同总压缩率下，比均匀分配秩的精度明显更好
```

### 量化兼容性优化（Hadamard 变换）

```
问题: SVD 低秩分解后的隐表示 H 存在严重异常值
  - SVD 将最大奇异值排在前几维 → 前几维数值极大
  - 这种分布对量化非常不友好（outlier 会拉大量化范围）

解决方案: Walsh-Hadamard 变换（WHT）
  原始: W ≈ A * B
  引入 WHT 矩阵 R:  W ≈ (A*R) * (R^T * B) = A_hat * B_hat
  
  R 是 Hadamard 矩阵（正交矩阵）
  → R^T * B 将异常值分散到所有维度
  → A*R 对应的隐表示分布更均匀，易于量化

关键优化: R 可以离线融合进 A 和 B 矩阵
  → 推理时零额外计算开销！
```

---

## 四、实验结果

### 分解粒度对比（50% 压缩率，Llama-2-7B）

```
Wikitext-2 困惑度（越低越好）:
  Baseline:  5.47
  J-LRD:     5.62 (+0.15)  ← 精度最好，FLOPs 最高
  G-LRD:     6.01 (+0.54)  ← 平衡方案
  M-LRD:     6.75 (+1.28)  ← 精度最差

Zero-shot 平均准确率:
  Baseline:  65.05%
  J-LRD:     64.82% (-0.23%)
  G-LRD:     62.61% (-2.44%)
  M-LRD:     57.79% (-7.26%)
```

### 与量化叠加（50% 低秩 + 量化，Llama-2-7B）

| 方法 | Bits | PPL (Wiki) | KV Cache (GB, 128K) | 压缩率 |
|------|------|-----------|---------------------|--------|
| Baseline | 16 | 5.12 | 64.0 | 1x |
| KVQuant | 3 | 5.35 | 12.0 | 81.25% |
| **PALU-50% + 3bit** | **3** | **5.33** | **8.4** | **86.87%** |
| KVQuant | 2 | 6.95 | 8.0 | 87.50% |
| **PALU-50% + 2bit** | **2** | **5.76** | **5.6** | **91.25%** |

→ PALU+2bit 比 KVQuant-2bit PPL 低 1.19，同时压缩率更高

### 延迟加速（Llama-2-7B, RTX 4090, 50% 压缩率）

```
RoPE 注意力模块（64K 输入）:
  PALU 单独:      1.89× 加速 vs FP16 基线
  PALU + 4bit:    2.91× 加速

非 RoPE 注意力（64K 输入）:
  PALU 单独:      2.20× 加速
  PALU + 4bit:    6.17× 加速（矩阵融合完全消除重建开销）

端到端解码（64K, RoPE 模型）:
  PALU + 4bit:    2.59× 加速（vs FP16 基线）
  KIVI-4bit:      1.78× 加速
  → PALU 显著优于 KIVI
```

### LongBench 长上下文评测

```
Mistral-7B, 30% 低秩压缩, 3bit 量化:
  平均压缩率: 7.59×
  平均准确率: 40.77% (基线: 42.54%)
  → 7.59× 压缩下准确率仅损失 1.77%

LongChat-7B, 30% + 3bit:
  平均压缩率: 7.59×
  平均准确率: 34.33% (基线: 35.45%)
```

---

## 五、核心启示与局限

### 核心启示

```
1. KV cache 压缩存在被忽视的第三维度——hidden dimension：
   量化压缩: 数值精度维度
   Token 淘汰: 序列长度维度
   PALU 低秩: hidden dimension 维度
   三者完全正交，可自由叠加

2. "缓存隐表示、按需重建"的设计范式：
   不缓存最终 KV，而是缓存中间投影结果
   用更少空间存储同样丰富的信息
   这与 MLA（DeepSeek-V2 的 Multi-head Latent Attention）思想相通

3. 矩阵融合是消除运行时开销的关键：
   离线预计算 B * W_o 的融合矩阵
   将重建开销从运行时转移到离线阶段
   使 PALU 在实际部署中真正提速而非减速

4. 分解粒度选择影响深远：
   G-LRD（group-head）是精度-效率帕累托最优点
   为什么: SVD 在更大矩阵上能更好地捕获主成分
   但 J-LRD 的重建开销太高，M-LRD 精度损失太大
```

### 局限

```
1. RoPE 模型无法完全消除重建开销：
   RoPE 的非线性性阻止了离线矩阵融合
   必须在线重建 Key → 引入额外延迟
   对于 Llama 家族主流模型影响较大

2. 短序列加速收益有限：
   序列长度 < 4K 时，PALU 几乎没有加速
   只在 KV cache 成为瓶颈的长序列场景才有显著收益

3. 需要校准数据：
   Fisher 信息计算需要 2048 条校准样本
   分布偏移可能影响秩分配的优化效果

4. 与 MLA 的关系：
   PALU 是 post-training 方案，适用于已有 MHA/GQA 模型
   MLA 需要重新预训练，但推理效率更高
   PALU 填补了现有模型低成本改造的空缺
```

---

*解读日期：2026-04-07*
