# DéjàVu 论文深度解读

**论文:** DéjàVu: KV-cache Streaming for Fast, Fault-tolerant Generative LLM Serving
**作者:** Foteini Strati, Sara McAllister, Amar Phanishayee, Jakub Tarnawski, Ana Klimovic (MSR Project Fiddle / ETH Zurich / Carnegie Mellon University / Microsoft Research)
**arXiv:** 2403.01876
**代码:** 未公开（Microsoft Research内部系统）

---

## 一、解决什么问题？

分布式LLM服务系统（使用流水线并行）面临三大痛点，DéjàVu逐一解决：

```
痛点1：Pipeline Bubble（流水线气泡）

  Prompt处理时间 >> Token生成时间（最多差100倍）
  OPT-13B  实测: Prompt处理 ≈ Token生成的 14×
  OPT-66B  实测: Prompt处理 ≈ Token生成的 28×
  BLOOM-176B 实测: 差距超过 100×

  在流水线并行中：
  Stage 1: P1 P2 P3 P4 [1A 2A ...] [1B 2B ...]
  Stage 2:    P1 P2 P3 P4 [1A ...] [1B ...]
  ...
  当Microbatch 3提前结束（early stopping）时，
  新Microbatch 5的Prompt处理进入Stage 1，
  导致Stage 4还在生成Token 3A时就要等待Stage 1
  → 大量灰色区域（GPU idle time = Bubble）

痛点2：KV Cache内存过度预分配（Overprovisioning）

  FasterTransformer等系统为所有在途Microbatch
  在GPU内存中预分配完整KV cache（按最大序列长度）
  而每个时刻只有一个Microbatch在某Stage上活跃
  → 其余(D-1)个Microbatch的KV cache白白占用GPU内存
  → 批次大小（batch size）被严重限制

痛点3：分布式故障恢复代价极高

  GPU失效 → 所有在途请求的KV cache全部丢失
  → 系统必须从头重新处理每个请求（重跑整个prompt）
  GPT2-1.5B 测试：故障后恢复导致端到端延迟增加 1.89×
  在流水线并行场景：一个Stage故障 → 整个pipeline停摆
```

**核心问题：如何在流水线并行LLM推理系统中，同时解决气泡、内存浪费和故障恢复这三大问题？**

---

## 二、核心方法/关键发现

### 三大解决方案概览

```
解决方案1：Prompt-Token Disaggregation（阶段分离）
  将Prompt处理分配给专用P-workers（Prompt Workers）
  将Token生成分配给专用T-workers（Token Workers）
  → 两类任务在不同机器上并行，消除流水线气泡

解决方案2：Microbatch KV Cache Swapping（微批次KV换入换出）
  只在GPU内存中保留当前正在处理的Microbatch的KV cache
  其余Microbatch的KV cache放在CPU内存
  按需Swap In/Out（GPU ↔ CPU）
  → GPU内存需求大幅降低，可以承载更大批次

解决方案3：KV Cache Replication（KV cache复制）
  每个Worker将KV cache实时流式复制给下一个Worker
  故障发生时，从最近的副本恢复，无需从头重算
  → 故障恢复延迟大幅降低
```

### 核心基础设施：DéjàVuLib

```
DéjàVuLib 是一个模块化KV cache流式传输库，
提供三层抽象原语：

底层（flush/fetch）:
  flush: 将连续KV cache块从GPU/CPU拷贝到指定目标
         本地用CUDA，远程用NCCL/MPI/Boost
  fetch: 从源端获取连续KV cache块

中层（scatter/gather）:
  scatter: 将非连续KV cache区域切分成连续块并传输
  gather:  从源端收集多个分散块合并

高层（stream_out/stream_in）:
  stream_out: 给定源Worker、KV cache和推理配置，
              自动确定每个KV块的正确目标（可能需要split/merge）
  stream_in:  给定目标Worker，自动找到源并接收
  → 屏蔽了pipeline深度、batch size、TP配置等差异
```

---

## 三、技术细节

### Prompt-Token分离的资源分配优化

```
给定D台机器，每台M GB内存，模型L层：

Prompt Pipeline深度 D_p 满足:
  D_p >= ceil[ L * (C_0 + W_0) / M ]
  其中 C_0 = 单层prompt KV cache内存，W_0 = 单层模型权重

Token Pipeline深度 D_t 满足:
  D_t >= L * W_0 / (M - L*(C_0 + K_0))
  其中 K_0 = 单token KV cache增量

最优分配（令Prompt Pipeline和Token Pipeline吞吐相等）:
  D_t = D * N * t / (m * Y + N * t)
  D_p = D - D_t = D * m * Y / (m * Y + N * t)

  其中:
    Y = 单个prompt处理时间（ms）
    t = 单个token生成时间（ms）
    N = 每个Microbatch生成的新token数
    m = KV cache流式传输开销系数（m ≥ 1，DéjàVuLib使m≈1）

disaggregated系统吞吐 > non-disaggregated 的条件:
  Y/t > (D-1) / (D*(2-m)-1)
→ 当 Y/t 大（prompt很慢）或 D 大时，disaggregation更有益
```

### KV Cache流式传输的三项关键优化

```
优化1：Buffered Copies（缓冲拷贝）
  Token生成时，KV cache更新是大量小的非连续内存写入
  (每个token只更新每层的一小条)
  直接调用多次cudaMemcpy → 极高overhead（131× slowdown）

  解决：在GPU DRAM中维护临时聚合Buffer
    每个token生成后 → 先写入GPU Buffer
    Buffer满后 → 一次性拷贝到CPU内存
  效果：95× 性能提升（相比朴素方法）

优化2：Layer-by-layer Prompt Cache Streaming（逐层流水线传输）
  Prompt处理是逐层进行的
  → 可以在处理第i层时，同时传输第i-1层的KV cache
  类似训练中的Wait-Free Backpropagation思路
  → Prompt KV cache传输延迟完全被计算掩盖

优化3：Token Streaming Parallelization（Token传输并行化）
  Token生成是多步骤的（Microbatch A生成第t步时，
  Microbatch B也在生成其对应步骤）
  → 将 Microbatch x 在第t步的KV cache传输，
    与 Microbatch x+1 在第t+1步的计算并行执行
  → Token传输延迟完全被计算掩盖

综合效果（DéjàVuLib vs 朴素流式传输）:
  GPT2-1.5B: 131× slowdown → 约 1× slowdown (几乎无开销)
  OPT-13B:    92× slowdown → 约 1× slowdown
  OPT-66B:    69× slowdown → 约 1× slowdown
  → DéjàVuLib流式传输延迟 < 2%（完全被计算掩盖）
```

### Microbatch KV Cache Swapping

```
流水线深度D，D个Microbatch同时在途：
  GPU内存分配: 2*M GB（仅当前+下一个Microbatch的KV cache）
  CPU内存分配: D*M GB（所有Microbatch的KV cache）

  vs 传统预分配: 需要 D*M GB GPU内存

每步Token生成时（Stage 4处理Microbatch x，生成token T1x）:
  1. Swap In: 将Microbatch x+1 的KV cache从CPU搬到GPU
  2. 计算 T1x
  3. Swap Out: 将Microbatch x 更新后的KV cache送回CPU
  → Swap In/Out与计算完全并行执行

关键约束: 搬运一个Microbatch的KV cache时间
          必须 < 一步token生成时间（10s~100ms级）
  PCIe带宽通常够用（16GB/s），但需要DéjàVuLib的3项优化支撑
```

### 故障检测与恢复流程

```
KV cache复制策略:
  Worker x 将KV cache实时流式复制给 Worker (x+1)%N
  （N个Worker构成环形复制链）

故障检测:
  Controller通过心跳检测Worker失效
  → 通知所有其他Worker停止服务

四步恢复流程（以Stage 2故障为例）:
  1. Stage 3 将 Stage 2 的副本KV cache复制回 Stage 2（重填）
  2. Stage 1 将自身KV cache复制给 Stage 2（恢复副本链）
  3. Controller找到需要重新执行的 microbatch j，step t
     (j=1, t=C: Stage 2在故障前KV cache复制到step C)
  4. Stage 1重启 microbatch j 从 step t 继续推理

故障恢复测试（OPT-66B，4台机器，pipeline深度4）:
  baseline: 故障导致延迟增加 1.91×
  DéjàVu:   故障导致延迟增加 1.24×
  → 故障恢复延迟减少约1.54×
```

---

## 四、实验结果

### 实验环境

```
硬件: 
  - 2块A100-80GB GPU的VM，inter-VM带宽40Gbps
  - 2块V100-16GB GPU的VM，inter-VM带宽32Gbps
模型: GPT2-1.5B, OPT-13B, OPT-66B, BLOOM-176B（HuggingFace版）
基线: FasterTransformer（支持Pipeline Parallelism的SOTA框架）
```

### 端到端性能（无故障）

```
OPT-66B (6台V100 vs 1台+3台 disaggregated):
  DéjàVu-1-3 vs Baseline-4:
  在 request rate ≤ 1.5 rps 时，DéjàVu 延迟更低
  在 1.88× 请求率下，DéjàVu 维持与 Baseline 相同延迟
  → 吞吐提升 1.88×

BLOOM-176B (7台A100 vs 2台+4台 disaggregated):
  DéjàVu-2-4 vs Baseline-7:
  在 request rate ≤ 0.6 rps 时，DéjàVu 延迟更低
  吞吐提升约 2×
```

### Microbatch Swapping 效果

```
OPT-30B (6台V100-16GB，batch=B vs 2B):
  无swap，batch=B:           ~0.10 req/s
  开启swap，batch=2B:        ~0.11 req/s（提升约10%）

OPT-66B (2台A100-80GB):
  无swap，batch=B:           ~0.08 req/s
  开启swap，batch=2B:        ~0.13 req/s（提升约65%）

BLOOM-176B (5台A100-80GB):
  无swap，batch=B:           ~0.05 req/s
  开启swap，batch=2B:        ~0.10 req/s（提升约100%）

→ 模型越大、GPU内存越紧张，swap收益越显著
```

### 故障场景性能（OPT-66B，4台机器）

```
无故障:         延迟曲线平稳
Baseline故障:   延迟从正常 → 跳升 1.91× → 重新处理
DéjàVu故障:    延迟从正常 → 跳升 1.24× → 快速恢复

多次故障（在600s、1200s、1800s各触发一次）:
  Baseline: 每次故障后所有请求重启，吞吐急剧下降
  DéjàVu:  从最近复制的step重启，运行时缩短约 1.16×
```

---

## 五、核心启示与局限

### 核心启示

```
1. KV cache的"流动性"是解锁多种优化的关键
   只要能高效流式传输KV cache（DéjàVuLib保证），
   就可以同时实现：
   · 阶段分离（Prompt→P-worker，Token→T-worker）
   · 内存优化（Microbatch级别Swap）
   · 故障恢复（异步复制+从最近checkpoint恢复）
   一个底层库解锁三个系统级优化

2. 非连续小内存更新是KV cache传输的主要障碍
   Token生成时每步只更新KV cache的一小块
   → 直接传输导致131×开销
   → 缓冲聚合（Buffered Copies）将问题规模转化
   这是许多系统忽略的底层细节

3. 流水线气泡的根本原因是任务异构
   Prompt（compute-bound）和Token（memory-bound）
   的计算特性完全不同，不应共享同一流水线资源
   → 专用资源分离是比调度优化更根本的解法

4. 故障恢复开销 ∝ 已生成token数
   传统方案必须重算所有token；
   DéjàVu只需从最近复制的step重算（少量token）
   → KV cache复制是"增量checkpoint"的思路
```

### 局限性

```
1. 仅针对流水线并行场景
   Tensor Parallelism（TP）场景下的气泡问题不同
   DéjàVu未处理TP场景（论文聚焦于PP）

2. Swap受限于PCIe带宽
   CPU-GPU KV cache传输依赖PCIe（通常16GB/s）
   当序列很长（>2K token）时，swap时间可能超过token生成时间
   → 需要高带宽互联（NVLink/CXL）才能发挥最大效果

3. 复制增加存储开销
   每个Worker额外存储邻居的KV cache副本
   对于BLOOM-176B等模型，这相当于显著增加内存压力

4. 资源分配规划是静态的
   D_p和D_t的分配在服务开始时确定
   无法动态适应workload变化（如prompt长度分布变化）

5. 与context caching不兼容
   DéjàVu的KV cache在请求结束后被丢弃
   未探索跨请求的历史KV cache复用（与MemServe不同）
```

### 在知识体系中的位置

```
分布式LLM服务系统演化路径:

  单机推理优化  → FasterTransformer（Pipeline PP基础设施）
  阶段分离探索  → DéjàVu（本文）/ Splitwise / DistServe
  统一内存抽象  → MemServe（在disaggregated架构上加caching）
  集群KV池化    → Infinite-LLM / Mooncake

  DéjàVu的贡献：
  ① 第一个将disaggregation用于解决PP架构的气泡问题
  ② 构建了高性能KV cache流式传输库DéjàVuLib
  ③ 首个在流式传输框架下解决故障恢复的系统

  与Splitwise/DistServe的区别:
  · Splitwise: 模拟为主，不考虑Pipeline Parallelism
  · DistServe: 关注goodput优化，基于模型特性分配资源
  · DéjàVu:   聚焦Pipeline PP + 故障容错，实际部署场景
```

---

*解读日期：2026-04-07*
