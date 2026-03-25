# FlashAttention 论文深度解读

**论文:** FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
**作者:** Tri Dao 等 (Stanford University)
**发表:** 2022年，被几乎所有现代LLM框架采用
**arXiv:** 2205.14135

---

## 一、解决什么问题？

Transformer的自注意力机制有两个致命问题：
- **时间复杂度**: O(N²) — 序列越长，计算量平方增长
- **内存复杂度**: O(N²) — 需要存储巨大的N×N注意力矩阵

之前的"近似注意力"方法（Linformer、Performer等）试图降低计算量，但**实际墙钟时间并没快多少**。为什么？

> **关键洞察：瓶颈不在计算(FLOPs)，而在内存访问(IO)！**

## 二、GPU内存层次结构 — 问题根源

```
┌─────────────────────────────────────────┐
│  SRAM (片上高速缓存)                      │
│  容量: 20MB    带宽: 19 TB/s   ← 超快！  │
├─────────────────────────────────────────┤
│  HBM (高带宽显存)                         │
│  容量: 40-80GB  带宽: 1.5 TB/s  ← 慢13倍 │
├─────────────────────────────────────────┤
│  CPU DRAM (主内存)                        │
│  容量: >1TB    带宽: 12.8 GB/s  ← 慢150倍│
└─────────────────────────────────────────┘
```

SRAM快13倍但只有HBM的1/4000容量。标准attention不断在HBM和SRAM之间搬运数据，这才是真正的瓶颈。

## 三、标准Attention为什么慢？

标准实现分4步，每步都要读写HBM：

```
步骤1: 从HBM加载Q,K → 计算 S = QKᵀ → 写S到HBM     (N×N矩阵!)
步骤2: 从HBM读S → 计算 P = softmax(S) → 写P到HBM   (又一个N×N矩阵!)
步骤3: 从HBM加载P,V → 计算 O = PV → 写O到HBM

总共读写HBM: Θ(Nd + N²) 次
存储中间矩阵: O(N²) 内存
```

N=2048时，S和P各占 2048×2048×2 bytes = 8MB。N=16K时变成 512MB。每一层attention都要存这两个矩阵！

## 四、FlashAttention核心算法

FlashAttention用两个关键技术解决问题：**分块(Tiling)** + **重计算(Recomputation)**

### 核心思想：永远不把N×N矩阵写入HBM

```
标准Attention:
  Q,K,V (HBM) → S=QKᵀ (写HBM) → P=softmax(S) (写HBM) → O=PV (写HBM)
                    ↑ N×N矩阵           ↑ N×N矩阵
                  必须存储！            必须存储！

FlashAttention:
  Q,K,V (HBM) → 分块加载到SRAM → 在SRAM内完成所有计算 → 只写最终O到HBM
                                    ↑
                          N×N矩阵从不出现在HBM中！
```

### 分块计算流程

```
外层循环: 遍历K,V的每个块 (j = 1 to Tc)
  │
  │  将 Kⱼ, Vⱼ 从HBM加载到SRAM
  │
  └─ 内层循环: 遍历Q的每个块 (i = 1 to Tr)
       │
       │  将 Qᵢ, Oᵢ 从HBM加载到SRAM
       │  在SRAM上计算: Sᵢⱼ = QᵢKⱼᵀ
       │  在SRAM上计算: softmax (增量式，维护running max和sum)
       │  在SRAM上更新: Oᵢ
       │  将 Oᵢ 写回HBM
       │
       └─ 完成
```

### 增量Softmax — 最巧妙的部分

标准softmax需要看完整行才能计算（需要全局max和sum）。FlashAttention用**在线算法**一块一块计算：

```
处理第1块后:  m₁ = max(S₁),  ℓ₁ = Σexp(S₁ - m₁)
处理第2块后:  m₂ = max(m₁, max(S₂))
              ℓ₂ = exp(m₁ - m₂)·ℓ₁ + Σexp(S₂ - m₂)
              O₂ = (ℓ₁·exp(m₁-m₂)·O₁ + exp(S₂-m₂)·V₂) / ℓ₂

→ 每次只需保存 m(max) 和 ℓ(sum) 两个标量，就能增量更新softmax
→ 结果与标准softmax数学上完全等价 — 这是精确计算，不是近似！
```

### 反向传播：重计算代替存储

```
标准方法: 前向时存储 S, P (O(N²)内存) → 反向时直接用
FlashAttention: 前向时只存 O 和 (m, ℓ) → 反向时重新计算 S, P

多用了一些FLOPs，但省了大量HBM读写 → 反而更快！
```

## 五、IO复杂度分析

| | HBM访问次数 | 内存占用 |
|---|---|---|
| **标准Attention** | Θ(Nd + N²) | O(N²) |
| **FlashAttention** | Θ(N²d²M⁻¹) | O(N) |

其中 M = SRAM大小(~20MB), d = head维度(64-128), N = 序列长度

当 d² << M（通常 64² = 4096 << 100K）时，FlashAttention的HBM访问比标准方法少很多倍。

论文还证明了一个**下界**：任何精确注意力算法都无法比 Θ(N²d²M⁻¹) 做得更好 — FlashAttention是**渐近最优的**。

## 六、性能结果

### 速度对比（A100 GPU）

| 实现 | GFLOPs | HBM读写(GB) | 运行时间(ms) |
|------|--------|------------|------------|
| 标准Attention | 66.6 | 41.7 | 41.7 |
| **FlashAttention** | 75.2 | **4.4** | **7.3** |

> FLOPs反而更多（因为重计算），但HBM访问减少了9.5倍，速度快了5.7倍

### 训练加速

| 模型 | vs HuggingFace | vs Megatron-LM |
|------|----------------|----------------|
| GPT-2 small | **3.5×** | **1.7×** |
| GPT-2 medium | **3.0×** | **1.8×** |
| BERT-large | 比MLPerf记录快**15%** | — |
| 长序列(1K-4K) | **2.4×** | — |

### 内存节省

```
标准Attention:  内存 ∝ N²  (序列长度的平方)
FlashAttention: 内存 ∝ N   (线性！)

实测：省 5-20× 内存
→ 同样GPU可以训练 4× 长的序列
→ GPT-2 从 1K上下文 → 4K上下文，perplexity还降了0.7
```

### 首次突破：超长序列任务

| 模型 | Path-X (16K) | Path-256 (64K) |
|------|-------------|----------------|
| 所有其他Transformer | ✗ (OOM或随机) | ✗ |
| **FlashAttention** | **61.4%** | ✗ |
| **Block-Sparse Flash** | 56.0% | **63.1%** |

FlashAttention是**第一个**能在Path-X上超过随机水平的Transformer。

## 七、与PagedAttention的关系

```
FlashAttention vs PagedAttention — 解决不同层面的问题：

┌──────────────────────────────────────────────┐
│            LLM推理系统                        │
│                                              │
│  ┌────────────────────────────────┐          │
│  │  PagedAttention (vLLM)         │          │
│  │  解决: KV cache在GPU内存中      │          │
│  │  如何分配、共享、回收           │          │
│  │  层面: 内存管理(请求间)         │          │
│  └────────────────────────────────┘          │
│                                              │
│  ┌────────────────────────────────┐          │
│  │  FlashAttention                │          │
│  │  解决: 单次attention计算        │          │
│  │  如何高效利用GPU内存层次        │          │
│  │  层面: 计算内核(单次计算)       │          │
│  └────────────────────────────────┘          │
│                                              │
│  两者互补，vLLM内部就使用FlashAttention       │
└──────────────────────────────────────────────┘
```

## 八、一句话总结

> **FlashAttention发现attention的真正瓶颈是内存IO而非计算量，通过分块计算+增量softmax+重计算，避免将N×N注意力矩阵写入GPU显存，实现了速度快2-4×、内存省5-20×的精确(非近似)注意力算法，成为所有现代LLM的底层基石。**

## 阅读建议

论文库中还有 FlashAttention2 (25_FlashAttention2.pdf)，是v2改进版，主要优化了并行性和work partitioning，建议接着读。

---

*解读日期：2026-03-25*
