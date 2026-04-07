# Quest 论文深度解读

**论文:** Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference
**作者:** Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, Song Han
**机构:** Shanghai Jiao Tong University, MIT, University of Washington, NVIDIA
**会议:** ICML 2024 (PMLR 235)
**arXiv:** 2406.10774
**代码:** https://github.com/mit-han-lab/Quest

---

## 一、解决什么问题？

随着LLM上下文窗口扩展到128K甚至1M token，长上下文推理的速度急剧下降：

```
问题根源: decode阶段每步都要加载完整KV cache

Llama-7B, 32K上下文:
  KV cache = 16 GB
  读取时间 = 11 ms/token
  → 占总推理时间 53% 以上！

GPU的推理正在被内存IO拖垮，而不是被计算拖垮
```

前人工作（H2O、TOVA、StreamingLLM）通过**历史注意力分数**来选择保留哪些KV，
但这些方法有根本性缺陷：

```
Query-Agnostic的问题:
  "A is B. C is D. A is"
                 ↑ 当前query="is"
  → 最后的"is"需要关注"B"
  → 但在之前的所有历史步骤中，"B"的注意力分数可能都很低
  → H2O会把"B"淘汰掉 → 答错！

核心洞察: token的重要性高度依赖当前query，不是静态的！
```

**Quest的目标：根据当前Query动态确定哪些KV是关键的，实现真正的Query-Aware稀疏性。**

---

## 二、核心方法/关键发现

### 发现1：注意力天然稀疏，但稀疏模式随Query变化

```
实验: LongChat-7B模型，32K上下文
  → 除前两层外，90%以上的层 sparsity > 90%
  → 理论上只需加载10%的KV cache即可

但不同的Query需要不同的KV子集：
  Query A → 关键token: {3, 7, 15, 42, ...}
  Query B → 关键token: {1, 8, 22, 56, ...}
  完全不同！固定策略必然失效。
```

### 发现2：可以用Key的Min/Max值近似估计关键性

Quest的核心思路：**不加载完整KV就能估计每个page的关键性**

```
对于每个KV cache页面 (包含S个token的keys):
  维护每个特征维度的:
    M_i = max(k_{1,i}, k_{2,i}, ..., k_{S,i})  ← 元素最大值
    m_i = min(k_{1,i}, k_{2,i}, ..., k_{S,i})  ← 元素最小值

给定当前Query向量 Q:
  对维度i，该页面的注意力分数上界:
    U_i = max(Q_i * M_i, Q_i * m_i)

  页面重要性分数 = sum(U_i)  ← 所有维度求和

→ 选Top-K个分数最高的页面，只加载这K个页面的完整KV进行计算
```

### Quest两阶段执行流程

```
阶段1: 估计页面重要性 (轻量级)
  输入: 当前Query + 每页的Min/Max元数据
  操作: 元素乘法 + 最大值 + 求和 → 每页一个分数
  开销: 只需加载元数据(2M*L/S字节)，远小于完整KV

阶段2: 稀疏Attention (只在重要页面上)
  输入: Top-K页面的完整KV + 当前Query
  操作: 标准Self-Attention
  加速: 只读1/S + K/N的总KV数据量

总数据量 = 原始的 (1/PageSize + K/PageNum)
```

---

## 三、技术细节

### 关键元数据维护

```python
# 插入新token到KV cache时增量更新:
for i in range(dim):
    M_i = max(M_i, k_i)   # 更新各维度最大值
    m_i = min(m_i, k_i)   # 更新各维度最小值

# Self-Attention时估计页面分数:
score = 0
for i in range(dim):
    score += max(q_i * max_i, q_i * min_i)
# → score是该页面最高注意力权重的上界
```

### 低稀疏度层的处理

```
观察: 前两层的稀疏度低于10% (不适合Quest优化)
处理: 前两层使用完整KV cache
     后续层使用Quest的稀疏Attention
→ 保证模型准确性不受影响
```

### CUDA内核实现

```
三个专用算子:
  1. Criticality Estimation: 元数据加载 + 分数计算
     开销 O(1/PageSize) → 随序列增长而相对减小
  
  2. Top-K Filtering: GPU并行TopK (借助RAFT向量搜索库)
     延迟 5-10μs，独立于序列长度
  
  3. Approximate Attention: 稀疏Attention计算
     延迟 = 常数 (取决于token budget K，不依赖总序列长度)

集成到 FlashInfer 推理框架
```

### 与PageAttention兼容

```
Quest以页为粒度管理KV cache (默认16 tokens/page)
→ 与PageAttention的内存管理完全兼容
→ 无需修改内存分配逻辑
```

---

## 四、实验结果

### 长依赖任务 (Passkey Retrieval)

```
10K token测试 (LongChat-7b-v1.5-32k):
┌────────────────┬──────────────────────────────────┐
│ 方法           │ 64   128   256   512  token预算  │
├────────────────┼──────────────────────────────────┤
│ H2O            │  0%   1%   1%    1%              │
│ TOVA           │  0%   1%   1%    3%              │
│ StreamingLLM   │  1%   1%   1%    3%              │
│ Quest (ours)   │ 65%  99%  99%   99%              │
└────────────────┴──────────────────────────────────┘

100K token测试 (Yarn-Llama-2-7b-128k):
  Quest-1024 token: 96% recall率  (H2O仅2%, TOVA 2%)
  Quest-2048 token: 100% recall率

→ Query-Aware让Quest在长依赖任务上碾压其他方法
```

### LongBench 综合评测

```
6个长文本数据集 (NarrativeQA, HotpotQA, GovReport等):
  Quest-1K token budget ≈ Full KV cache 性能
  其他方法即使4K token budget仍有明显差距

无损精度所需的token预算对比 (NarrativeQA, 平均24K上下文):
  Full KV:  24723 tokens
  TOVA:     14101 tokens  ← 仍需14K
  Quest:     5120 tokens  ← 只需5K，效率提升 4.5×
```

### 自注意力延迟加速

```
序列长度32K, token budget 2048:
  FlashInfer (基准):    ~ 650 μs
  Quest-2048:           ~  90 μs
  加速比:               7.03×

端到端延迟 (32K context, 4-bit量化):
  FlashInfer FP16:  36.8 ms/token
  Quest FP16:       22.4 ms/token  → 1.74× 加速
  Quest 4bit-AWQ:   29.6 ms/token  → 2.23× 加速

GovReport任务同等精度下的延迟比较:
  TOVA延迟: 315 μs
  Quest延迟:  71 μs  → 3.82× 更快
```

---

## 五、核心启示与局限

### 核心启示

```
1. Query-Aware vs Query-Agnostic 的关键差异:
   - 以前的方法用"历史"来预测未来 → 对长依赖任务失效
   - Quest用"当前Query"来决定谁重要 → 本质上更准确

2. 近似上界足够好:
   - Min/Max元数据是注意力分数的严格上界
   - 实验证明与Oracle稀疏性高度吻合
   - 低成本元数据 → 高质量的页面选择

3. 内存IO是瓶颈，不是计算:
   - 长上下文推理是内存受限的(memory-bound)
   - 降低内存访问量 = 直接加速，比算法优化更有效

4. 页粒度比token粒度更实用:
   - 细粒度(per-token)选择需要随机IO → 效率低
   - 粗粒度(per-page)选择适合批量IO → GPU友好
```

### 局限性

```
1. 前两层无法优化:
   - 前两层稀疏度低(<10%)，Quest在这里不适用
   - 对于有大量低层计算的模型，收益降低

2. 近似误差:
   - Min/Max上界有时过于保守，选入不必要的页面
   - 实际精度略低于Oracle稀疏性

3. 额外内存开销:
   - 需要存储每页的Min/Max元数据
   - 约占总KV cache的 2/PageSize ≈ 12.5% (PageSize=16)

4. 尚未支持prefill阶段优化:
   - 当前Quest只优化decode阶段
   - prefill阶段仍是全量注意力计算
```

---

## 六、在知识体系中的位置

```
KV Cache管理方法的演进:

  StreamingLLM (固定策略)
  → 保留前几个sink token + 最近token
  → 优点: 极简; 缺点: 丢失中间所有信息

  H2O (历史统计)
  → 基于历史累积注意力分数淘汰
  → 优点: 有理论保证; 缺点: 忽视query依赖性

  TOVA (当前状态)
  → 基于当前注意力分数永久淘汰
  → 缺点: 同样会永久丢弃将来可能重要的token

  Quest (Query-Aware)
  → 保留全部KV，按需选择加载哪些
  → 核心: 不淘汰，而是智能选择
  → 最适合: 长文本推理、需要访问远程token的任务
```

## 一句话总结

> **Quest发现token的关键性高度依赖当前Query而非历史统计，提出用每页KV的Min/Max元数据来近似估计注意力得分上界，动态选择Top-K关键页面进行稀疏Attention，实现7.03×自注意力加速和2.23×端到端延迟降低，在长依赖检索任务上以极小token预算（1%）达到100%召回率，从根本上超越了所有基于历史信息的KV淘汰方法。**

---

*解读日期：2026-04-07*
