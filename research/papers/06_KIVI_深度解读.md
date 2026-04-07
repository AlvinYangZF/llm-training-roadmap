# KIVI 论文深度解读

**论文:** KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache
**作者:** Zirui Liu*, Jiayi Yuan*, Hongye Jin, Shaochen (Henry) Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, Xia Hu
**机构:** Rice University, Texas A&M University, Stevens Institute of Technology, Carnegie Mellon University
**会议:** ICML 2024
**arXiv:** 2402.02750
**代码:** https://github.com/jy-yuan/KIVI

---

## 一、解决什么问题？

```
KV Cache 的内存困境:

  OPT-175B, batch=512, prompt=512, output=32:
  KV Cache = 512 × (512+32) × 层数 × hidden_dim × 2 × 精度字节
           = 约 1.2 TB  (是模型参数的 3.8×！)

  → 单A100 80GB根本放不下
  → 必须减小KV Cache才能增大batch size
  → batch size大 → GPU利用率高 → 推理速度快
```

**关键问题：** 现有工作简单地对KV Cache做per-token量化，直接量化到2bit精度会损失严重。但为什么呢？

```
已有方案的问题:
  INT4 round-to-nearest (per-token): 基本可用，但到INT2就崩了
  INT2 per-channel (Key和Value都用): 精度严重下降
  INT2 per-token (Key和Value都用): 精度严重下降

KIVI的核心发现:
  Key 和 Value cache 需要沿不同维度量化！
  → Key: per-channel 量化
  → Value: per-token 量化
  这个非对称设计是KIVI的核心创新
```

---

## 二、核心方法/关键发现

### 关键发现1：Key Cache 有固定的离群通道

```
Key Cache 的分布可视化 (Llama-2-13B, Layer 16):

  绝对值
    ▲
  12│      ██
  10│      ██        ██
   8│      ██        ██
   6│  ██  ██    ██  ██
   4│  ██  ██  ██████████
   2│████████████████████████...
   0└──────────────────────────→ Channel (维度)

  → 少数固定的Channel (列)具有极大数值 → "离群通道"
  → 离群通道在所有token中持续存在（固定模式）
  → 每个Channel内部的值彼此相近
  → 结论: Key应该 per-channel 量化
    (同一Channel内的值组合在一起，误差限制在各自Channel内)
```

### 关键发现2：Value Cache 没有通道离群值

```
Value Cache 的分布 (Llama-2-13B, Layer 16):

  绝对值
    ▲
   2│ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓
   1│ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓
   0└──────────────────────────────────────→ Channel

  → 没有明显的通道离群值模式
  → 但value cache用于加权求和attention输出
  → 单个token的量化误差不应影响其他token
  → 结论: Value应该 per-token 量化
    (同一token的所有channel组合在一起)
```

### 关键发现3：为何per-token比per-channel对Value更好？

```
理论分析:

  Attention输出 = Σ_j A_ij × V_j    (j遍历所有token)

  其中 A_ij 是attention scores (高度稀疏!)
  → 输出实际上是少数"重要token"的Value的加权和

  per-token量化: 每个token j的误差是独立的
    → 量化误差只影响该token自身
    → 不重要的token有误差 → 影响小 (因为A_ij小)
    → 重要的token误差小 → 整体影响小

  per-channel量化: 某channel c的误差影响所有token
    → 该channel在所有重要token上都有误差
    → 误差被attention scores放大传播
    → 对输出影响更大！

量化误差对比 (Llama-2-13B):
  Value per-token: Avg Δ = 3.55  (相对误差)
  Value per-channel: Avg Δ = 49.89  ← 14× 更大！
```

---

## 三、技术细节

### KIVI 算法设计

```
量化公式:
  Q(X) = floor((X - z_X) / s_X)
  X' = Q(X) · s_X + z_X

  其中:
    z_X = min(X)         (零点，asymmetric)
    s_X = (max(X) - min(X)) / (2^B - 1)  (缩放因子)
    B   = 量化位数 (2 for KIVI)

Key量化 (per-channel, group-wise):
  每 G 个token组成一个group (G=32)
  对每个channel维度上的G个token一起量化
  → scaling factor s ∈ R^{l_prompt}  (每token一个)

Value量化 (per-token):
  每个token单独量化其所有channel
  → scaling factor s ∈ R^d  (每channel一个)
```

### 流式KV Cache的挑战与解决方案

```
挑战: KV Cache是流式数据结构（token逐个到达）

  per-token量化对Value: 直接可行！
    → 新token的Value直接量化后append到现有cache

  per-channel量化对Key: 不能直接流式处理！
    → channel的缩放因子需要跨所有token计算
    → 新token到来时，之前已量化的key会失效

解决方案: Key Cache 分两部分
  ┌───────────────────────────────────────────────┐
  │        Key Cache 总体结构                       │
  ├──────────────────────┬────────────────────────┤
  │  Grouped Key Cache   │  Residual Key Cache    │
  │  X_{K_g}             │  X_{K_r}               │
  │  (已完成整组，量化为2bit)│ (最新R个token，全精度)  │
  ├──────────────────────┴────────────────────────┤
  │                   R ≤ 128                      │
  └───────────────────────────────────────────────┘

  每 R 个新token到来:
    1. 将 X_{K_r} 中 R 个token量化 → 追加到 X_{K_g}
    2. 重置 X_{K_r} 为空
    3. 新token进入新的 X_{K_r}

  类似地，Value Cache也有 Residual:
    X_{V_r}: 最新R个token的Value，全精度保留
    X_{V_g}: 之前的Value，量化为2bit
```

### Decoding 阶段的注意力计算

```
解码时的矩阵乘法（Tiled Matrix Multiplication）:

  Raw attention logits:
    A_g = t_Q · Q(X_{K_g})^T     (量化部分，混合精度MatMul)
    X_{K_r} = Concat([X_{K_r}, t_K])  (追加新key)
    A_r = t_Q · X_{K_r}^T          (残差部分，全精度)
    A = Concat([A_g, A_r])          (拼接)

  关键: Q_MatMul 融合反量化 + 矩阵乘法
    → 量化KV不需要先完全反量化再计算
    → 在 tiling 层级融合 → 节省显存带宽

硬件友好实现:
  → CUDA实现 Q_MatMul kernel
  → Triton实现分组量化 kernel
  → 与vLLM完全兼容（插件式使用）
```

---

## 四、实验结果

### 量化方案对比（伪量化实验）

```
Llama-2-13B, group_size=32, 2bit:

方案                    CoQA    TruthfulQA
16bit (全精度)          66.37   29.53
4bit (K-T, V-T)         66.48   29.51  ← 4bit几乎无损
2bit (K-C, V-T)         63.53   28.60  ← KIVI方案，最优！
2bit (K-T, V-T)         52.93   24.98  ← per-token Key：差
2bit (K-C, V-C)          2.88    0.74  ← per-channel Value：崩了
2bit (K-T, V-C)          2.80    0.26  ← 两个都per-channel：崩了

结论: Key per-channel + Value per-token 是2bit的唯一可行方案
```

### 主要模型的准确率 (KIVI-2 vs 16bit)

```
模型        CoQA    TruthfulQA   GSM8K
Llama-2-7B:
  16bit     63.88    30.76       13.50
  KIVI-2    63.05    33.95       12.74  ← 几乎无损！

Llama-2-13B:
  16bit     66.37    29.53       22.67
  KIVI-2    66.23    29.84       20.77  ← 轻微下降

Mistral-7B:
  16bit     67.40    30.45       38.36
  KIVI-2    66.35    32.17       36.01  ← 几乎无损！

注: Falcon-7B使用multi-query attention(MQA)，
    单head的KV已高度压缩，需要KIVI-4才能保持精度
```

### LongBench 长文本测试

```
KIVI-2 在 LongBench 上的平均性能:

模型             16bit   KIVI-4  KIVI-2
Llama2-7B        44.52   44.59   44.27  ← 仅 -0.3%
Llama2-13B       44.85   44.48   44.69  ← 仅 -0.2%
Llama2-7B-Chat   45.96   45.83   45.67  ← 仅 -0.3%
Mistral-7B       46.58   46.56   45.85  ← -0.7%

Needle-in-a-Haystack (20K words):
  KIVI-2 仍能在长文本中检索到正确信息
  与全精度基线几乎没有差异
```

### 吞吐量与内存效率

```
峰值内存 (Llama-2-7B, A100 80GB):
  16bit baseline:         ~60 GB (batch≈100时OOM)
  KIVI-2 (residual=128):  ~35 GB (batch≈300时仍可运行)
  KIVI-2 (residual=32):   ~30 GB (batch≈400时仍可运行)
  → 2.6× 峰值内存节省

吞吐量 (tokens/s, Llama-2-7B, A100):
  16bit:          ~500 tok/s (batch=100)
  KIVI-2 (r=128): ~1500 tok/s (batch=300)
  KIVI-2 (r=32):  ~2500 tok/s (batch=400)
  → 2.35× ~ 3.47× 吞吐量提升

内存节省 = 更大batch = 更高GPU利用率 = 更高吞吐量
```

---

## 五、核心启示与局限

### 核心启示

```
1. KV Cache的Key和Value有根本不同的统计特性
   → Key: 存在固定通道离群值，应per-channel量化
   → Value: 无明显通道离群值，应per-token量化
   → 非对称设计是正确答案，强行对称会导致精度崩溃

2. Residual Cache的设计使流式量化成为可能
   → Key per-channel量化需要批量处理token
   → 保留最新R个token为全精度解决了流式不兼容性
   → R不能太大（增加内存开销），建议≤128

3. 量化位数并非越低越好，需要考虑架构兼容性
   → MQA/GQA架构（Falcon/Mistral变体）的KV head数少
   → 已高度压缩的KV用KIVI-2会损失过多精度
   → MHA架构模型用KIVI-2更安全

4. 2bit量化是实用极限（结合Residual策略）
   → 纯2bit（无Residual）在GSM8K等硬任务上有明显下降
   → 保留全精度Residual是在2bit下维持性能的关键
```

### 局限性

```
1. 对MQA/GQA架构效果下降
   → Falcon-7B（单head KV）需要KIVI-4而非KIVI-2
   → 随着Grouped Query Attention普及，适用性受限

2. Prefill阶段量化开销较大
   → 目前Prefill阶段量化是额外开销
   → 论文提到可与前序运算融合，但留为未来工作

3. 量化带来的精度损失在推理型任务上更明显
   → GSM8K（数学推理）有~1-2%精度损失
   → 对精度敏感任务需谨慎使用KIVI-2，可考虑KIVI-4

4. group_size=32 是超参数，需要验证
   → group_size=128时性能明显下降
   → 实际部署需要针对每个模型调整
```

---

## 六、一句话总结

> **KIVI通过对LLM的KV Cache分布进行系统研究，发现Key cache存在固定通道离群值而Value cache不存在，据此提出非对称2bit量化方案（Key按通道量化，Value按token量化），并设计了Residual Cache机制解决流式量化的工程挑战，实现Llama/Falcon/Mistral模型2.6×内存节省，几乎无精度损失，带来2.35×~3.47×推理吞吐量提升，是首个在2bit精度下实现免微调KV Cache量化的实用方案。**

---

*解读日期：2026-04-07*
