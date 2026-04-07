# FlashAttention-2 论文深度解读

**论文:** FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
**作者:** Tri Dao (Princeton University / Stanford University)
**会议/期刊:** arXiv 2023, ICLR 2024
**arXiv:** 2307.08691
**代码:** https://github.com/Dao-AILab/flash-attention

---

## 一、解决什么问题？

注意力计算是Transformer扩展到长上下文的核心瓶颈：

```
标准Attention的内存问题:
  S = QK^T  →  形状 N×N, 内存 O(N^2)
  P = softmax(S)
  O = PV

  N=8192 时: S和P各占 8192^2 × 2bytes ≈ 128MB
  → 每层attention需要写入/读取 256MB 到HBM
  → 这些读写远多于实际的矩阵乘法计算量
```

FlashAttention-1 已解决了内存问题（IO-Awareness），但：

```
FlashAttention-1 的不足:
  - 前向传播: 只达到理论最大TFLOPs/s的 30-50%
  - 反向传播: 只达到理论最大TFLOPs/s的 25-35%
  - GEMM操作可以达到 80-90%

原因: GPU上的工作划分不优化
  (1) 非matmul FLOPs过多
  (2) 序列长度维度没有并行化 → GPU占用率低
  (3) warp间的工作划分引入不必要的shared memory读写
```

**核心问题：如何在保持FlashAttention正确性的同时，让attention计算达到接近GEMM的硬件利用率？**

---

## 二、核心方法/关键发现

FlashAttention-2 提出三项关键改进：

### 改进1：减少非matmul FLOPs

```
背景: GPU的matmul吞吐量远高于非matmul操作
  A100: matmul峰值 312 TFLOPs/s (FP16)
        非matmul峰值  19.5 TFLOPs/s (FP32)
  → 每个非matmul FLOP相当于16个matmul FLOP的时间代价

FlashAttention-1 的代价:
  每步更新输出时需要对diag(l^(2))^(-1) 重缩放两次

FlashAttention-2 的优化:
  改为维护"未缩放"的中间输出 O_tilde
  只在最后一步做一次缩放:
  
  O_tilde^(j) = diag(e^(m^(j-1)-m^(j)))^(-1) * O_tilde^(j-1) + e^(S^(j)-m^(j)) * V^(j)
  
  最终输出: O = diag(l^(last))^(-1) * O_tilde^(last)
  
  额外优化: 只存储 logsumexp L = m + log(l)
  而不是分别存储 m 和 l (用于反向传播)
```

### 改进2：序列长度维度并行化

```
FlashAttention-1的并行策略:
  并行维度: batch × head
  → 每个thread block处理一个attention head
  → 总thread blocks = batch_size × num_heads
  
  问题: 当序列很长时, batch_size × num_heads 可能 < 108 (A100的SM数)
  → GPU利用率不足

FlashAttention-2的改进:
  在序列长度维度也并行化!
  
  前向传播:
  ┌────────────────────────────────────────┐
  │ 外循环按Q的行块并行 → 每个worker处理一块行 │
  │ 内循环串行扫描K,V的列块 (无需通信)       │
  └────────────────────────────────────────┘
  
  反向传播:
  ┌────────────────────────────────────────┐
  │ 按K,V的列块并行                         │
  │ 不同thread blocks用atomic adds更新dQ   │
  └────────────────────────────────────────┘
  
  → 即使batch_size=1, num_heads=1也能充分利用GPU
```

### 改进3：优化warp间工作划分

```
FlashAttention-1 的"split-K"方案 (前向传播):
  K,V 按列分给4个warp
  每个warp计算 Q*K^T 的一部分
  → 结果写入shared memory → 同步 → 累加
  → 大量shared memory读写!

FlashAttention-2 的改进方案:
  Q 按行分给4个warp (K,V所有warp共享)
  每个warp计算自己slice of Q 对应的输出行
  → 无需warp间通信!
  → 大幅减少shared memory读写
  
  图示:
  FlashAttention-1:     FlashAttention-2:
  Q: [warp1-4共享]      Q: [w1][w2][w3][w4]
  K: [w1][w2][w3][w4]   K: [warp1-4共享]
  V: [w1][w2][w3][w4]   V: [warp1-4共享]
  → 结果需要归约          → 每个warp独立完成
```

---

## 三、技术细节

### 完整前向传播算法

```
Algorithm: FlashAttention-2 Forward Pass
输入: Q, K, V ∈ R^(N×d) in HBM, block sizes Bc, Br

1. 将Q分成 Tr = ceil(N/Br) 个块: Q_1,...,Q_Tr (每个 Br×d)
   将K,V分成 Tc = ceil(N/Bc) 个块: K_1,...,K_Tc (每个 Bc×d)

2. for i = 1 to Tr:
     从HBM加载Q_i到SRAM
     初始化 O_i = 0, l_i = 0, m_i = -inf
     
     for j = 1 to Tc:
       从HBM加载K_j, V_j到SRAM
       S_i^(j) = Q_i * K_j^T              # 在chip上计算
       m_i^(j) = max(m_i^(j-1), rowmax(S_i^(j)))
       P_tilde_i^(j) = exp(S_i^(j) - m_i^(j))
       l_i^(j) = e^(m_i^(j-1) - m_i^(j)) * l_i^(j-1) + rowsum(P_tilde_i^(j))
       O_tilde_i^(j) = diag(e^(m_i^(j-1)-m_i^(j)))^(-1) * O_tilde_i^(j-1) + P_tilde_i^(j)*V_j
     
     O_i = diag(l_i^(Tc))^(-1) * O_tilde_i^(Tc)
     L_i = m_i^(Tc) + log(l_i^(Tc))      # 存logsumexp
     将O_i, L_i写回HBM
```

### 因果掩码优化

```
Causal Mask处理:
  对于列索引 > 行索引的块: 整块跳过 (结果全是-inf → attention=0)
  → 大约跳过一半的块 (序列很长时)
  → 额外1.7-1.8× 加速 (有causal mask vs 无)

每行只需对1个块应用causal mask (恰好在对角线上的块)
其余块要么全计算, 要么全跳过
```

### Block size选择

```
典型block sizes:
  {64, 128} × {64, 128}  (由head dim d和GPU shared memory决定)

权衡:
  更大的block → 更少的shared memory reads/stores
              → 但需要更多registers和shared memory
              → 可能导致register spilling或无法运行

针对每个head dimension手动调优 (4种选择)
```

---

## 四、实验结果

### Attention速度基准 (A100 80GB SXM4)

```
前向传播 (head dim=128, 无causal mask):
  序列长度:    512    1k     2k     4k     8k    16k
  PyTorch:     42     48     48     48     48    OOM
  FlashAttn:  103    121    122    122    122    122  TFLOPs/s
  xformers:    60     63     63     63     63     63
  FlashAttn2: 163    187    196    198    200    203  TFLOPs/s
  → FA2是FA1的 1.58×, 是PyTorch的 3.6×+

前向+反向 (head dim=128, 有causal mask):
  序列长度:    2k     4k     8k    16k
  FlashAttn:   81     91     92     92  TFLOPs/s
  FlashAttn2: 221    265    294    308  TFLOPs/s (H100)
  → H100上FA2达到 73% 理论峰值 (前向)
  → H100上FA2达到 63% 理论峰值 (前向+反向)
```

### 端到端训练速度 (8×A100 80GB)

```
GPT-style模型训练:

  Model          Context  w/o FA   FA1      FA2
  GPT3-1.3B      2k       142      189      196  TFLOPs/s/GPU
  GPT3-1.3B      8k        72      170      220  TFLOPs/s/GPU
  GPT3-2.7B      2k       149      189      205  TFLOPs/s/GPU
  GPT3-2.7B      8k        80      175      225  TFLOPs/s/GPU

→ FA2 vs FA1: 最高 1.3× 加速
→ FA2 vs 无FA: 最高 2.8× 加速
→ FA2在GPT3-2.7B 8k上达到 225 TFLOPs/s (72% model FLOPs利用率)
```

### H100 GPU上的表现

```
运行相同代码 (不做H100特殊优化):
  前向+反向速度可达 335 TFLOPs/s
  → 使用TMA和4th-gen Tensor Cores预计还能额外1.5-2×提升
```

---

## 五、核心启示与局限

### 核心启示

```
1. IO-Awareness之外, 工作划分同样关键
   → FlashAttention-1解决了"读写什么", FA-2解决了"谁来做"
   → 非matmul FLOPs看起来少, 实际代价是matmul的16倍
   
2. 序列长度维度的并行化打破了batch×head的限制
   → 对于长上下文推理 (batch_size小) 尤为重要
   → 是后续支持128K+上下文的硬件基础

3. warp级工作划分设计决定了shared memory通信开销
   → "split-Q" vs "split-K": 消除warp间通信
   → 这一思路也影响了后续的FlashAttention-3等工作

4. 算法正确性不变 (精确计算, 无近似)
   → 训练、微调、推理均可无缝替换
   → 已成为所有主流框架的默认实现

5. FA2直接支持MQA和GQA
   → 通过在backward中对KV梯度求和实现
   → 为现代高效模型架构提供硬件支撑
```

### 局限性

```
1. Block size需要手动调优
   → 目前针对4种head dimension手动选择
   → 作者提出未来用auto-tuning解决

2. H100新特性尚未利用
   → TMA (Tensor Memory Accelerator)
   → 4th-gen Tensor Cores
   → 预计可获得额外1.5-2× 提升 (已在FA3中实现)

3. 反向传播实现更复杂
   → 反向需要5个matmul (前向只需2个)
   → 多了dQ需要通过atomic adds跨thread block累加的开销

4. 仅针对标准Attention
   → 复杂的Attention变体 (如稀疏Attention) 需要额外工作
   → FlashInfer等后续工作解决了这一问题
```

### 在技术生态中的位置

```
Attention优化演进:
  FlashAttention-1 (2022): IO-Aware tiling, 无近似, 2-4× 加速
         ↓
  FlashAttention-2 (2023): 更好的并行化+工作划分, 再2× 加速  ← 本文
         ↓
  FlashAttention-3 (2024): H100 TMA/Tensor Core特化, 进一步提升
         ↓
  FlashInfer (2025):       推理专用, 可定制变体, 动态调度
```

---

> **FlashAttention-2通过三项精准优化——减少非matmul FLOPs、在序列维度并行化、以"split-Q"替换"split-K"消除warp通信——在FlashAttention-1基础上再获约2×加速，使attention计算在A100上达到73%理论峰值TFLOPs，端到端训练速度提升至225 TFLOPs/s，成为当前所有主流LLM训练与推理框架的标准底层内核。**

---

*解读日期：2026-04-07*
