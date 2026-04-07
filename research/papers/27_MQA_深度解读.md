# MQA (Multi-Query Attention) 论文深度解读

**论文:** Fast Transformer Decoding: One Write-Head is All You Need
**作者:** Noam Shazeer (Google)
**会议/期刊:** arXiv 2019
**arXiv:** 1911.02150
**代码:** 无官方代码 (思路简单, 已被广泛复现)

---

## 一、解决什么问题？

自回归Transformer解码的速度瓶颈不在计算，而在**内存带宽**：

```
Transformer增量推理 (每生成一个token):
  
  必须从内存加载的数据:
  ┌─────────────────────────────────────────────────────┐
  │  1. 模型权重 W_q, W_k, W_v, W_o (每层)              │
  │  2. KV Cache: 所有历史token的K和V (尺寸随序列增长)   │
  └─────────────────────────────────────────────────────┘
  
  数学分析 (假设 h个head, 每head维度 d_k=d_v=d/h):
  
  每n步推理的总内存访问 (n为序列长度, b为batch size):
    Θ(bn²d + bnd²)
    第1项: 来自K和V的访问   ← 问题所在!
    第2项: 来自投影矩阵
  
  内存访问 vs 计算量的比值:
    Θ(n/d + 1/b)
    → 当 n≈d 或 b≈1 时, 比值接近1
    → 内存带宽成为主要瓶颈!
```

**核心问题：KV cache的大小与head数量成正比，而大量head意味着海量内存访问。能否在保留多个query head的同时，减少key/value head的数量？**

```
MHA的KV cache尺寸 (每层, 每个token):
  K: [h, d_k]  →  h×d_k 个元素
  V: [h, d_v]  →  h×d_v 个元素
  
  典型设置 (h=8, d_k=d_v=128):
  每token每层 KV = 2 × 8 × 128 × 2 bytes = 4096 bytes = 4KB
  
  序列长度n=1024, 32层模型, batch=128:
  KV cache = 4KB × 1024 × 32 × 128 = 16 GB!
  → 在内存中反复读写这些数据成为瓶颈
```

---

## 二、核心方法/关键发现

### Multi-Query Attention (MQA) 的核心思想

```
MHA (原始):
  每个head有独立的 K, V 投影矩阵 P_k^(i), P_v^(i)
  
  K^(i) = M * P_k^(i)   形状 [b, m, d_k]  (每个head一份)
  V^(i) = M * P_v^(i)   形状 [b, m, d_v]  (每个head一份)
  
  KV cache尺寸: [b, h, m, d_k] + [b, h, m, d_v]
  随head数h线性增长

MQA (本文提出):
  所有query head共享同一组 K, V 投影矩阵!
  
  K = M * P_k    形状 [b, m, d_k]  (所有head共用一份!)
  V = M * P_v    形状 [b, m, d_v]  (所有head共用一份!)
  Q^(i) = x * P_q^(i)  (每个head各自的query投影)
  
  KV cache尺寸: [b, 1, m, d_k] + [b, 1, m, d_v]
  → KV cache缩小 h 倍 (h为原来的head数)!
```

### 内存带宽分析

```
MQA增量推理的内存访问 (n步推理, batch=b):

  总内存访问: Θ(bnd + bn²k + nd²)
  其中:
    bnd  → x, q, o, y 相关的访问
    bn²k → K和V的访问 (现在只有1个head!)
    nd²  → 投影矩阵 P_q, P_k, P_v, P_o

  内存/计算 比值:
    Θ(1/d + n/dh + 1/b)

  对比MHA的比值: Θ(n/d + 1/b)

  → 将 n/d 项缩小了 h 倍!
  → 理论上应大幅改善内存带宽瓶颈
```

### 代码层面的极简改动

```python
# MHA (标准多头注意力) - 批处理版
def MultiheadAttentionBatched(X, M, mask, P_q, P_k, P_v, P_o):
    # P_k: shape [h, d, k]  ← 每个head各一份
    # P_v: shape [h, d, v]  ← 每个head各一份
    Q = einsum("bnd,hdk->bnhk", X, P_q)
    K = einsum("bmd,hdk->bmhk", M, P_k)
    V = einsum("bmd,hdv->bmhv", M, P_v)
    logits = einsum("bnhk,bmhk->bnhm", Q, K)
    ...

# MQA (多查询注意力) - 仅移除K,V的"h"维度
def MultiqueryAttentionBatched(X, M, mask, P_q, P_k, P_v, P_o):
    # P_k: shape [d, k]  ← 所有head共用!
    # P_v: shape [d, v]  ← 所有head共用!
    Q = einsum("bnd,hdk->bnhk", X, P_q)   # Q依然是多头的
    K = einsum("bmd,dk->bmk",   M, P_k)   # K只有1个head
    V = einsum("bmd,dv->bmv",   M, P_v)   # V只有1个head
    logits = einsum("bnhk,bmk->bnhm", Q, K)  # 注意: K无h维度
    ...
```

---

## 三、技术细节

### 增量推理中的KV cache对比

```
MHA增量推理 (每步):
  prev_K: [b, h, m, k]  ← h份, 内存占用大
  prev_V: [b, h, m, v]
  
  新token的K: einsum("bd,hdk->bhk", x, P_k)
  新K: concat([prev_K, new_K], axis=2)  → [b, h, m+1, k]

MQA增量推理 (每步):
  prev_K: [b, m, k]     ← 只有1份, 内存占用小 h倍!
  prev_V: [b, m, v]
  
  新token的K: einsum("bd,dk->bk", x, P_k)
  新K: concat([prev_K, new_K], axis=2)  → [b, m+1, k]
  
  内存访问减少: h倍 (h=8时减少8倍的KV加载)
```

### 模型容量保持技巧

```
问题: 去掉K,V的head维度 → 参数量减少 → 模型容量下降?

解决方案: 扩大Feed-Forward层的宽度
  MHA baseline:     d_ff = 4096, 总参数 211M
  MQA实验模型:      d_ff = 5440  (扩宽), 总参数仍为 211M
  
  → 将KV投影节省的参数"补偿"到FFN中
  → 使对比在相同总参数量下进行
```

### 实验设置

```
任务1: WMT 2014 英德翻译
  基线: 6层Encoder-Decoder Transformer
  d_model=1024, d_ff=4096, h=8, d_k=d_v=128
  211M参数, 100K步训练, 32核TPUv3

任务2: Billion-Word Language Modeling Benchmark
  6层decoder-only Transformer
  d_model=1024, d_ff=8192, h=8, d_k=d_v=128
  192M参数, 136K步训练, 32核TPUv3
```

---

## 四、实验结果

### 模型质量 (WMT14 EN-DE翻译)

```
模型类型          h    d_k,d_v  d_ff   ln(PPL)  BLEU(dev)  BLEU(test)
multi-head        8    128      4096   1.424    26.7       27.7/28.4
multi-query       8    128      5440   1.439    26.5       27.5/28.5  ← 最高!
multi-head local  8    128      4096   1.427    26.6       27.5/28.3
multi-head        1    128      6784   1.518    25.8       26.8/27.9
multi-head        2    64       6784   1.480    26.2       (未报告)
multi-head        4    32       6784   1.488    26.1       (未报告)
multi-head        8    16       6784   1.513    25.8       (未报告)

关键发现:
  → MQA质量略低于MHA (PPL: 1.439 vs 1.424)
  → 但MQA显著优于减少head数/维度的替代方案
  → MQA beam-4解码达到28.5 BLEU, 甚至超过MHA的28.4!
```

### 推理速度 (TPUv2, 序列长度128)

```
指标: 每输出token的摊销推理时间 (微秒)

模型类型         训练      推理                  Beam-4搜索
                        enc+dec              enc+dec
multi-head       13.2    1.7 + 46 μs/tok     2.0 + 203 μs/tok
multi-query      13.0    1.5 + 3.8 μs/tok    1.6 + 32  μs/tok  ← 快12×!
multi-head local 13.2    1.7 + 23 μs/tok     1.9 + 47  μs/tok
multi-query local 13.0   1.5 + 3.3 μs/tok    1.6 + 16  μs/tok  ← 最快!

关键数字:
  MHA解码器: 46μs/token
  MQA解码器:  3.8μs/token
  → 解码速度提升 12× (beam=1)
  → Beam-4搜索提升 6× (203 → 32μs)
  → 训练速度几乎不变 (13.2 vs 13.0μs)
```

### Billion-Word LM基准

```
模型类型      h    d_k,d_v  d_ff   dev-PPL
multi-head    8    128      8192   29.9    ← 最优
multi-query   8    128      9088   30.2    ← 仅轻微下降
multi-head    1    128      9984   31.2
multi-head    2    64       9984   31.1
multi-head    4    32       9984   31.0
multi-head    8    16       9984   30.9

→ MQA (30.2) 优于所有减少head数/维度的MHA变体
→ 与full MHA的质量差距远小于其他替代方案
```

---

## 五、核心启示与局限

### 核心启示

```
1. 推理瓶颈是内存带宽, 不是计算量
   → 对于增量解码, n/d 项占主导
   → 减少KV cache大小 h 倍 → 解码速度提升 ~h 倍
   → 这是近年LLM推理优化的核心指导原则之一

2. "一个write-head就够了" — 极简设计的威力
   → 仅改变K,V的投影矩阵, 无需其他架构改动
   → 代码变更极小 (einsum中去掉一个"h"维度)
   → 却带来超过10倍的解码加速

3. 参数效率的重新分配
   → 节省的KV参数"补偿"到FFN → 维持总参数量
   → 模型容量损失极小, 而推理效率大幅提升

4. 为MQA变体奠定基础
   → MQA是GQA (Grouped-Query Attention) 的特例 (G=1)
   → Llama2/3, Mistral, Falcon等现代LLM均使用GQA
   → MQA的思路直接启发了整个KV cache压缩方向

5. 训练和推理可以针对性优化
   → MQA训练速度几乎与MHA相同
   → 专门为推理性能设计架构是合理的权衡
```

### 局限性

```
1. 质量轻微下降
   → MQA的PPL通常比MHA略高
   → 在需要极高质量的任务上可能有影响
   → GQA (2023) 通过引入G>1个KV head解决了这个问题

2. MQA训练不稳定
   → 从头训练MQA可能出现loss spikes
   → 在长输入任务的fine-tuning时可能发散
   → GQA通过"uptraining"策略（从MHA检查点转换）更加稳定

3. 只评估encoder-decoder模型
   → 2019年decoder-only架构尚未主导
   → 后续工作证明MQA在decoder-only LLM中同样有效
   → 且decoder-only没有cross-attention, MQA优势更为直接

4. 更适合大batch/长序列
   → 当 n << d 且 b >> 1 时, 内存带宽不是瓶颈
   → MQA的优势在小batch、长序列、增量解码场景最为显著
```

### 在技术演进中的位置

```
注意力机制的KV head演进:

  MHA (2017, Vaswani et al.)
  h个query head, h个KV head
  KV cache: O(h × n × d_k)
       ↓
  MQA (2019, Shazeer)           ← 本文
  h个query head, 1个KV head
  KV cache: O(1 × n × d_k)
  → 解码速度 ~h×, 质量轻微下降
       ↓
  GQA (2023, Ainslie et al.)
  h个query head, G个KV head (1<G<h)
  KV cache: O(G × n × d_k)
  → 质量接近MHA, 速度接近MQA
  → Llama3, Mistral, Gemma等主流模型采用
```

---

> **MQA通过让所有query head共享单一组KV投影，将增量推理中的KV cache加载量缩减h倍，在保持几乎相同参数量和轻微质量损失的前提下，实现了解码速度12倍提升；这一"一个write-head就够了"的极简洞见奠定了现代LLM高效推理架构的基础，并直接催生了被广泛采用的Grouped-Query Attention。**

---

*解读日期：2026-04-07*
