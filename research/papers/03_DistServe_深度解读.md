# DistServe 论文深度解读

**论文:** DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving
**作者:** Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang
**机构:** 北京大学、StepFun、UC San Diego
**会议:** OSDI 2024
**arXiv:** 2401.09670
**代码:** https://github.com/LLMServe/DistServe

---

## 一、解决什么问题？

LLM推理服务有两个截然不同的阶段，但现有系统将它们混在一起：

```
现有系统问题：Prefill + Decode 共置在同一GPU

  Prefill 阶段特性：
  ├─ 计算密集（compute-bound）
  ├─ 一次处理整个prompt（数百~数千token）
  └─ 优化目标: 降低 TTFT (Time To First Token)

  Decode 阶段特性：
  ├─ 内存带宽密集（memory-bound）
  ├─ 每次只生成一个token
  └─ 优化目标: 降低 TPOT (Time Per Output Token)

  两者共置 → 严重的 Prefill-Decode 互相干扰！
  ├─ Prefill拖慢Decode → TPOT上升
  └─ Decode阻塞Prefill → TTFT上升
  → 为满足严格SLO必须过度配置GPU资源 → 成本高昂
```

**核心问题：** 如何在满足TTFT和TPOT双重SLO约束的前提下，最大化每GPU的goodput（吞吐量）？

---

## 二、核心方法/关键发现

### 核心思路：彻底分离 Prefill 和 Decode

```
DistServe 架构：

  ┌─────────────────────────────────────────────────┐
  │                   Controller                     │
  └──────────────┬────────────────────┬─────────────┘
                 │                    │
        ┌────────▼────────┐  ┌────────▼────────┐
        │  Prefill Instance│  │  Decode Instance│
        │   (专用GPU组)    │  │   (专用GPU组)   │
        │                 │  │                 │
        │  LLM Model      │  │  LLM Model      │
        │  GPU | GPU      │  │  GPU | GPU      │
        │  Parallel RT    │  │  Parallel RT    │
        └────────┬────────┘  └────────▲────────┘
                 │    KV Cache        │
                 └────────────────────┘
                    传输KV Cache

→ 每个阶段独立优化自己的并行策略和资源分配
→ 消除互相干扰，各自聚焦TTFT或TPOT目标
```

### 关键发现1：两阶段的并行策略偏好不同

```
Prefill 阶段（低负载时）:
  intra-op并行 > inter-op并行
  原因: 降低单次执行时间 → 直接降低TTFT

  平均TTFT (2-way inter-op并行):
  Avg_TTFT_inter = D + RD²/(4-2RD)

  平均TTFT (2-way intra-op并行, 加速比K):
  Avg_TTFT_intra = D/K + RD²/(2K(K-RD))
  → 低负载下 D/K 项主导，intra-op更优

Decode 阶段（大batch时）:
  inter-op并行 > intra-op并行
  原因: 线性扩展吞吐量，满足TPOT SLO
```

### 关键发现2：KV Cache 传输开销可忽略

```
KV Cache 传输开销分析 (OPT-66B, 512 token请求):
  单请求KV Cache大小 ≈ 1.13 GB
  10 rps时需传输带宽 ≈ 11.3 GB/s = 90 Gbps

  现代GPU集群配置:
  ├─ InfiniBand: 800 Gbps (跨节点)
  └─ NVLINK: 600 GB/s (节点内)
  → 传输开销可忽略不计！

  关键设计: Decode实例主动"拉取(pull)" KV Cache
  而非Prefill实例推送 → 利用Prefill GPU内存作缓冲区
```

---

## 三、技术细节

### 放置算法（Placement Algorithm）

**高节点亲和集群（High Node-Affinity Cluster, 有InfiniBand）：**

```
Algorithm 1: 两阶段独立优化

输入: LLM G, 节点限制N, 每节点GPU数M, 内存容量C, 工作负载W, 流量率R

for intra_op in {1..M}:
  for inter_op in {1..N×M/intra_op}:
    if G.size / (inter_op × intra_op) < C:
      config ← (inter_op, intra_op)
      G_hat ← parallel(G, config)
      模拟Prefill goodput → 选最优 config_p
      模拟Decode goodput  → 选最优 config_d

n,m ← ⌈R/config_p.goodput⌉, ⌈R/config_d.goodput⌉
best_plm ← (n, config_p, m, config_d)
```

**低节点亲和集群（Low Node-Affinity Cluster，无高速跨节点网络）：**

```
Algorithm 2: 联合优化Prefill+Decode段
  核心约束: Prefill和Decode的对应层必须在同一节点内
  → 强制KV Cache通过NVLINK传输
  → 将实例切分为"instance segments"，同一层的段共置
```

### 在线调度优化

```
1. 减少流水线气泡:
   → 对Prefill按prompt长度分组，总长接近L_m（计算饱和阈值）
   → 对Decode，batch_size = L_m

2. 应对突发流量:
   → Decode实例主动pull KV Cache（非push）
   → Prefill GPU内存作为天然缓冲队列

3. 动态重规划:
   → 工作负载分布器监控平均输入/输出长度、到达率
   → 检测到显著变化时触发重新运行放置算法（~1.3分钟）
```

---

## 四、实验结果

### 测试配置

```
集群: 4节点 × 8 NVIDIA A100-80GB (共32 GPU)
跨节点带宽: 25 Gbps（低亲和集群）
基准系统: vLLM、DeepSpeed-MII
测试应用: Chatbot (OPT-13B/66B/175B)、代码补全 (OPT-66B)、文档摘要 (OPT-66B)
```

### 主要性能数据

| 应用场景 | 模型 | vs vLLM | vs DeepSpeed-MII |
|---------|------|---------|-----------------|
| Chatbot | OPT-13B | 2.0~4.6× 更高rate | 1.6~7.4× 更高rate |
| Chatbot | OPT-66B | 2.0~4.6× 更高rate | 1.6~7.4× 更高rate |
| Chatbot | OPT-175B | 显著提升 | DeepSpeed不支持 |
| 代码补全 | OPT-66B | 大幅提升 | 明显提升 |

```
综合结论：
  → 最多服务 7.4× 更多请求 (vs 最优baseline)
  → SLO 约束最多收紧 12.6×
  → 90%+ 请求满足延迟要求

核心原因：
  分离消除Prefill-Decode互相干扰
  + 各阶段独立使用最优并行策略
  = 每GPU goodput大幅提升
```

### 175B 模型放置案例

```
DistServe为OPT-175B (ShareGPT数据集) 自动找到:
  Prefill: inter-op=3, intra-op=3  (9 GPU)
  Decode:  inter-op=3, intra-op=4  (12 GPU)

非对称配置！人工很难发现。
→ 验证了自动优化算法的价值
```

---

## 五、核心启示与局限

### 核心启示

```
1. 两阶段的根本差异决定了"共置"是系统性缺陷，不是工程问题
   → Prefill是compute-bound批处理作业
   → Decode是memory-bound延迟敏感任务
   → 强行共置必然相互干扰

2. 分离带来两个独立维度的优化空间
   → Prefill和Decode可以独立选择并行策略
   → 可以独立根据流量动态扩缩容

3. KV Cache传输是分离的唯一额外开销，但现代网络使之可忽略
   → NVLINK 600GB/s vs 传输需求 ~10GB/s
   → 开销 < 总时间的 7%

4. per-GPU goodput 是比吞吐量更好的优化指标
   → 同时考虑SLO达标率和成本效率
```

### 局限性

```
1. 故障容忍：Prefill与Decode实例耦合
   → 单个Decode实例故障可能波及多个Prefill实例
   → 需要更复杂的容错设计（论文留为未来工作）

2. Prefill请求长度不均匀导致流水线气泡
   → inter-op并行时不同长度请求使各阶段执行时间不同
   → 论文通过调度优化部分缓解，但未完全解决

3. 当前实现：内存管理不支持分页
   → 未集成PagedAttention等高级内存管理
   → 实际部署中与vLLM等系统的内存效率有差距

4. 不支持抢占（Preemption）
   → 可能出现"护送效应"（convoy effect）
   → 长Prefill请求会阻塞短请求
```

---

## 六、在知识体系中的位置

```
LLM推理优化全景：

  层面1 — 计算内核优化
  └→ FlashAttention: 单次attention计算加速

  层面2 — KV Cache压缩
  ├→ H2O: 保留重要KV，丢弃无用KV
  └→ KIVI/KVQuant: 量化压缩KV

  层面3 — 单机内存管理
  └→ PagedAttention/vLLM: 分页管理消除碎片

  层面4 — 集群级系统架构   ← DistServe 在此层面
  ├→ DistServe: Prefill/Decode分离 + 独立优化
  ├→ Splitwise: 同类方法，更关注硬件异构性
  └→ TetriInfer: 更细粒度的调度优化
```

---

## 七、一句话总结

> **DistServe发现LLM推理中Prefill（计算密集）和Decode（带宽密集）的根本差异使二者共置必然相互干扰，提出将两阶段分配到不同GPU并为每阶段独立优化并行策略与资源分配，通过自动化放置算法最大化per-GPU goodput，在真实工作负载下实现最高7.4×吞吐量提升或12.6×更严格的SLO约束，开创了LLM推理服务"解耦架构"的研究方向。**

---

*解读日期：2026-04-07*
