# Splitwise 论文深度解读

**论文:** Splitwise: Efficient Generative LLM Inference Using Phase Splitting
**作者:** Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Iñigo Goiri, Saeed Maleki, Ricardo Bianchini
**机构:** University of Washington, Microsoft
**会议:** ISCA 2024
**arXiv:** 2311.18677
**代码:** https://github.com/Azure/AzurePublicDataset (生产traces公开)

---

## 一、解决什么问题？

```
LLM推理两阶段的计算特性截然相反，共置在同一机器上效率低下：

  Prompt计算阶段 (Prefill):
  ├─ 计算密集 (compute-bound)
  ├─ GPU利用率高，接近TDP（热设计功耗）上限
  ├─ 吞吐量在 ~2048 tokens时饱和后下降
  └─ 需要最新H100等高算力GPU

  Token生成阶段 (Decode):
  ├─ 内存带宽密集 (memory-bound)
  ├─ GPU功耗 << TDP，大量算力浪费
  ├─ 吞吐量随batch size持续增长直到内存耗尽
  └─ 不需要高算力GPU，A100足够！

关键洞察（来自Azure生产traces）:
  Coding服务: 中位输入 1500 tokens, 中位输出 13 tokens
  对话服务: 中位输入 1020 tokens, 中位输出 129 tokens
  → 两种服务的输入/输出比差异极大
  → 混合批处理时大量时间浪费在少token的decode上
  → 60-70%的时间batch中只有≤20个token在运行
```

**核心问题：** 如何将两个阶段分配到最适合各自特性的硬件上，最大化成本效率（Perf/$）和功耗效率（Perf/W）？

---

## 二、核心方法/关键发现

### 方法核心：Phase Splitting（阶段分裂）

```
Splitwise 集群架构：

  ┌─────────────────────────────────────────────────┐
  │         Cluster-Level Scheduler (CLS)  (1)       │
  └──────┬──────────────────────────┬───────────────┘
         │                          │
  ┌──────▼──────┐  Mixed   ┌────────▼────────┐
  │ Prompt Pool │◄─────────►  Token Pool     │
  │  (3)        │  Pool    │  (3)            │
  │             │          │                 │
  │ Machine-    │  (5)KV   │ Machine-        │
  │ Level Sched │─cache ──►│ Level Sched     │
  │ (2)         │  transfer│ (2)             │
  │             │          │                 │
  │ Pending     │          │ Pending         │
  │ Queue  (4)  │          │ Queue  (4)      │
  └─────────────┘          └─────────────────┘

  关键: KV Cache 从 Prompt 机器传输到 Token 机器
  实现: MSCCL++ (Microsoft优化的GPU通信库)
  优化: 层级异步传输 (layer-wise async transfer)
```

### 关键发现：不同GPU适合不同阶段

```
A100 vs H100 对比 (来自实验数据):

              A100      H100    Ratio
  TFLOPs      19.5      66.9    3.43×
  HBM容量     80GB      80GB    1.00×
  HBM带宽     2039GBps  3352GBps 1.64×
  功耗         400W      700W    1.75×
  NVLink      50Gbps   100Gbps  2.00×
  成本         $17.6/hr  $38/hr  2.16×

  Prompt阶段 (compute-bound):
  → H100性能 3.43× 而成本仅 2.16× → H100划算

  Token阶段 (memory-bound):
  → H100带宽仅 1.64× 而成本 2.16× → A100更划算！

核心洞察7: Token生成可以用低算力硬件运行获得更好Perf/W 和 Perf/$
```

---

## 三、技术细节

### 集群级调度 (CLS)

```
三种机器池管理:
  Prompt Pool:  专门处理Prompt计算
  Token Pool:   专门处理Token生成
  Mixed Pool:   动态扩缩容的缓冲区 (负载偏离预期时使用)

请求路由: 使用 Join-the-Shortest-Queue (JSQ) 算法
  → 同时分配一对Prompt+Token机器
  → 可以重叠KV-cache传输与prompt计算
  → 减少传输开销

再利用机制:
  当 Prompt 机器队列过长 → 从Token池借调机器
  当 Token 机器队列过长 → 从Prompt池借调机器
  Mixed池中机器完成任务后归还原池
```

### KV Cache 传输优化

```
朴素方案 (Serialized):
  Prompt完成 → 开始传KV Cache → 传完 → Token开始解码
  → 传输延迟完全串行，阻塞第二个token生成

优化方案 (Layer-wise async):
  Layer 1计算完 → 异步传输Layer 1的KV Cache
  Layer 2开始计算  ← 同时进行
  Layer 2计算完 → 异步传输Layer 2的KV Cache
  ...
  → 传输与计算并行，大幅降低感知延迟

  对小prompt: 使用串行传输 (KV Cache小，无需层级传输)
  对大prompt: 使用层级传输 (掩盖大部分传输延迟)

实测结果:
  串行传输延迟随prompt增长线性增加，达到KV传输时间
  层级传输: 非重叠传输时间约 8ms (A100) 和 5ms (H100)
  → 层级传输几乎完全隐藏传输开销
```

### 机器级调度 (MLS)

```
Prompt机器:
  策略: FCFS (先来先服务)
  批处理限制: 总tokens ≤ 2048 (超过此阈值吞吐量下降)

Token机器:
  策略: FCFS + 内存感知
  批处理: 持续扩大batch直到内存耗尽
  内存跟踪: 实时监控KV Cache内存使用

Mixed机器:
  同时运行Prompt和Token任务
  Prompt任务优先级 > Token任务 (保证TTFT SLO)
  Token任务被抢占时增加优先级 (防饥饿)
```

### 四种 Splitwise 变体

```
Splitwise-AA: A100 (Prompt) + A100 (Token)  同构，基准
Splitwise-HH: H100 (Prompt) + H100 (Token)  同构，高性能
Splitwise-HA: H100 (Prompt) + A100 (Token)  异构，推荐！
Splitwise-HHcap: H100 (Prompt) + H100限功率70% (Token)

命名规则: 第一字母=Prompt机器, 第二字母=Token机器
```

---

## 四、实验结果

### 测试设置

```
数据集: Microsoft Azure生产traces (Coding + Conversation)
模型: BLOOM-176B, Llama2-70B (8× H100 DGX节点)
基准: Baseline-A100 (40台DGX-A100), Baseline-H100 (40台DGX-H100)
目标: 等功耗(iso-power)下最大吞吐量对比
```

### Iso-Power 吞吐量优化结果

```
对话场景 (Conversation trace), 等功耗比较:

  Splitwise-HH (25P, 15T):
  → 2.15× 吞吐量 vs Baseline-A100 (同功耗同成本)

  Splitwise-HA (25P, 15T):
  → 1.18× 吞吐量 vs Baseline-H100，节省10%成本，功耗不变

  Splitwise-HHcap (25P, 21T):
  → 同吞吐量 vs Baseline-H100，节省25%功耗，成本相同

编程场景 (Coding trace):
  Splitwise-AA (55P, 15T): 吞吐量 ≈ Baseline-A100
  Splitwise-HHcap: 明显优于所有基准
```

### KV Cache 传输开销分析

```
KV Cache 传输 vs 端到端延迟:
  串行传输: 最大约为E2E延迟的 3%
  层级传输: 用户感知影响 < 0.8%

  实测: 仅在第二个token延迟有所体现 (16.5% overhead)
  vs 串行传输的 64% overhead

结论: 传输开销对用户感知几乎不可见
```

---

## 五、核心启示与局限

### 核心启示

```
1. 硬件异构性是降本增效的关键杠杆
   → Prompt需要高计算力 (H100)
   → Token只需要高内存带宽 (A100足够)
   → 混合使用可同时优化吞吐量、成本、功耗

2. 生产traces揭示了被忽视的效率浪费
   → 混合批处理下GPU大部分时间空转
   → 60~70%时间只处理 ≤20 个token
   → 分离后Token机器可以维持高批处理度

3. 分层KV Cache传输几乎完全消除分离的唯一代价
   → 层级异步传输 + InfiniBand高带宽
   → 传输开销对用户感知 < 0.8%

4. 功耗上限(power cap)是新的优化维度
   → Token阶段内存密集不吃算力
   → 限制Token机器功耗到70% → 不损性能
   → 对数据中心CSP极具价值
```

### 局限性

```
1. KV Cache传输是瓶颈之一
   → 大prompt的KV Cache可能数GB
   → 跨节点传输依赖InfiniBand带宽
   → 低带宽环境下优势会减弱

2. 集群级调度器可能成为规模扩展瓶颈
   → 大规模集群（数千机器）时CLS开销增大
   → 论文建议参考分区调度等方案但未实现

3. 负载预测依赖历史分布
   → Prompt和Token池比例基于历史traces
   → 突发流量或分布偏移时Mixed Pool成为缓冲
   → 但频繁重配置(re-purposing)有开销

4. 不涉及KV Cache压缩
   → 传输的是全精度KV Cache
   → 与量化结合有进一步优化空间
```

---

## 六、与DistServe的比较

```
DistServe (OSDI 2024) vs Splitwise (ISCA 2024):

相同点:
  ├─ 都提出将Prefill和Decode分离到不同GPU
  ├─ 都通过网络传输KV Cache
  └─ 都实现了各阶段独立的资源和并行优化

不同点:
  DistServe:
    ├─ 关注点: goodput最大化，SLO达标率
    ├─ 放置算法: 自动搜索最优并行配置
    └─ 评估: 单cluster，模型至175B

  Splitwise:
    ├─ 关注点: 硬件异构，成本/功耗效率
    ├─ 创新: 不同GPU型号适合不同阶段
    └─ 评估: 使用真实Azure生产traces

结论: 两篇论文独立发现相同核心思路，各有侧重
```

---

## 七、一句话总结

> **Splitwise基于Microsoft Azure生产traces的深入分析发现LLM推理两阶段的计算和功耗特性截然相反，提出将Prompt计算（算力密集）与Token生成（带宽密集）分别部署在H100和A100等不同硬件上，通过层级异步KV Cache传输消除分离代价，实现等功耗下1.4×更高吞吐量（20%更低成本），或等吞吐量下25%更低功耗，为LLM推理集群的异构硬件设计提供了系统性方法论。**

---

*解读日期：2026-04-07*
