# Mooncake 论文深度解读

**论文:** Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving
**作者:** Ruoyu Qin 等 (Moonshot AI, 清华大学)
**会议:** USENIX FAST 2025 最佳论文
**arXiv:** 2407.00079
**实际部署:** Kimi (月之暗面) 生产系统

---

## 一、解决什么问题？

前面三篇论文解决的是**单机/单次计算层面**的问题。Mooncake解决的是**生产级集群部署**的系统架构问题：

> 当你有数百块GPU、数百万用户、输入长度从几百到十几万不等时，如何设计一个满足SLA、最大化吞吐量的LLM服务系统？

### 核心挑战

```
Prefill (预填充) 和 Decode (解码) 的特性完全不同：

         Prefill                    Decode
瓶颈      计算密集 (compute-bound)    内存密集 (memory-bound)
并行性    高 (所有prompt token并行)    低 (逐token串行)
SLO指标   TTFT (首token延迟)         TBT (token间延迟)
KV Cache  生产者 (计算KV)             消费者 (使用KV)
资源需求   大量算力,少量内存           少量算力,大量内存

→ 把它们放在同一组GPU上 = 互相干扰、资源浪费
```

## 二、核心架构：以KV Cache为中心的存算分离

```
┌─────────────────────────────────────────────────┐
│           Conductor (全局调度器)                    │
│  ┌──────────────┬──────────────┬────────────────┐│
│  │Cache-aware   │ KVCache      │ Load-balance   ││
│  │Prefill       │ Balance      │ Decoding       ││
│  │Scheduler     │ Scheduler    │ Scheduler      ││
│  └──────────────┴──────────────┴────────────────┘│
└─────────────────────────────────────────────────┘
         │                              │
    ┌────▼────┐                   ┌────▼────┐
    │ Prefill │  KVCache Transfer │ Decode  │
    │  Pool   │ ←───────────────→ │  Pool   │
    ├─────────┤   (RDMA/Messenger)├─────────┤
    │GPU/VRAM │                   │GPU/VRAM │
    │CPU/DRAM │← Distributed  →  │CPU/DRAM │
    │   SSD   │   KVCache Pool    │   SSD   │
    └─────────┘                   └─────────┘
```

**三大创新点：**

1. **Prefill和Decode物理分离** — 不同GPU集群各司其职
2. **分布式KV Cache池** — 利用CPU DRAM和SSD构建集群级缓存
3. **KV Cache感知的调度** — 调度决策围绕缓存命中率优化

## 三、请求处理全流程

```
用户请求到达
    │
    ▼
┌─ Conductor (全局调度器) ─────────────────────────┐
│ 1. 对请求token分块，计算每块的hash                    │
│ 2. 在所有Prefill节点中查找前缀缓存匹配               │
│ 3. 选择最优Prefill节点 (缓存命中最多 + 队列最短)       │
│ 4. 预选Decode节点 (负载均衡 + VRAM容量满足TBT SLO)   │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─ Step 1: KVCache Reuse (缓存复用) ──────────────┐
│ Prefill节点从分布式缓存池加载已有的前缀KV Cache      │
│ → 跳过已缓存部分的计算，只算新增部分                  │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─ Step 2: Incremental Prefill (增量预填充) ───────┐
│ 只计算未缓存token的KV Cache                       │
│ 长输入: 分chunk跨多节点pipeline并行 (CPP)          │
│ 逐层异步: 第L层算完→立刻存CPU DRAM→开始传输         │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─ Step 3: KVCache Transfer (缓存传输) ───────────┐
│ 通过Messenger组件 (GPUDirect RDMA)               │
│ 将完整KV Cache从Prefill节点流式传到Decode节点       │
│ 与Prefill计算重叠执行，减少等待                     │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─ Step 4: Decoding (解码生成) ────────────────────┐
│ Decode节点从CPU DRAM异步加载KV Cache到GPU VRAM     │
│ 加入continuous batching，逐token生成               │
└─────────────────────────────────────────────────┘
```

## 四、关键技术创新

### 4.1 分布式KV Cache池

```
传统: KV Cache只存在GPU VRAM中 → 容量极其有限
Mooncake: 构建3层存储层次

  GPU VRAM  (最快, 最小)  → 当前活跃请求的KV
     ↕
  CPU DRAM  (快, 较大)    → 前缀缓存 + 传输缓冲
     ↕
  SSD       (慢, 最大)    → 冷数据存储

KV Cache以paged block存储，每个block附带hash值
相同hash = 相同前缀 → 可复用！
```

### 4.2 Chunked Pipeline Parallelism (CPP)

长上下文输入（如128K tokens）怎么快速prefill？

```
传统方案: Sequence Parallelism (SP)
  → 切分序列到多节点，但每层需要all-reduce通信 → 网络开销大

Mooncake的CPP:
  将输入分成多个chunk，像pipeline parallelism一样处理
  → 只在chunk边界通信（不是每层）
  → 天然适配不同长度输入
  → 短输入: 单节点TP处理
  → 长输入: 多节点CPP pipeline处理
```

### 4.3 Layer-wise Prefill (逐层预填充)

```
普通prefill: 所有层算完 → 一次性存KV到DRAM → 一次性传输
                                               ↑ VRAM占用时间长

Layer-wise: 第L层算完 → 立即异步存DRAM → 开始传输 → 同时算第L+1层
                                               ↑ 计算和传输重叠！

效果: KV Cache存储延迟几乎被计算时间完全掩盖
      Prefill节点不需要为KV Cache预留VRAM
```

### 4.4 KVCache-centric调度算法

```
传统调度: 看哪个节点最空闲，就把请求发过去
Mooncake:  看哪个节点缓存命中最多，综合考虑：

  对每个请求R:
    1. 计算R的token块hash序列
    2. 在每个Prefill节点查找前缀匹配长度
    3. 找到缓存匹配最长的节点 (best_matched_instance)
    4. 估算TTFT = 队列等待时间 + 传输时间 + 计算时间
    5. 选TTFT最小的方案
    6. 如果最优方案不在best_matched节点 → 考虑迁移热点缓存
    7. 预选Decode节点 → 检查TBT SLO是否满足
    8. 如果都不满足SLO → 拒绝请求（early rejection）
```

### 4.5 过载场景的早期拒绝

Mooncake面对的现实：**GPU供不应求，必须拒绝部分请求**

```
朴素拒绝: 直接按当前负载拒绝 → 导致负载震荡（拒绝太多→突然空闲→又放太多→又过载）

Mooncake的预测式拒绝:
  1. 预测每个请求的输出长度（基于历史统计）
  2. 预测短期未来负载
  3. 优先拒绝那些"做了prefill但decode时会超SLO"的请求
  4. 分优先级调度

→ 避免"做了昂贵的prefill却最终拒绝"的浪费
```

## 五、性能结果

### 吞吐量提升

| 场景 | vs 基线 | 条件 |
|------|--------|------|
| 模拟长上下文 | **最高525%** | 满足TTFT/TBT SLO |
| Kimi真实流量 | **多处理75%请求** | 相同GPU资源 |

### 缓存命中率

| 缓存策略 | 1K blocks | 10K blocks | 50K blocks |
|---------|-----------|------------|------------|
| LRU | 0.30 | 0.40 | 0.50 |
| LFU | 0.30 | 0.35 | 0.49 |
| LengthAware | 0.30 | 0.35 | 0.48 |

### 关键数据

- 平均输入长度: 7,590 tokens
- 平均输出长度: 182 tokens
- 输入输出比: ~720:1（输入远长于输出 → prefill优化极其重要）
- 缓存块热度: 50%的块几乎不被访问，少数热点块被访问数万次

## 六、在知识体系中的位置

```
论文1: PagedAttention  — 单机GPU上如何管理KV cache内存
论文2: FlashAttention   — 单次attention如何高效计算
论文3: StreamingLLM     — 单个请求如何无限长推理
论文4: Mooncake        — 集群级别如何设计整个服务架构
                          ↑
                    把前面所有优化整合到生产系统中

技术栈关系:
  FlashAttention → 被Mooncake的prefill/decode内核使用
  PagedAttention → Mooncake的paged KV Cache池基于此思想
  StreamingLLM   → Mooncake的缓存淘汰策略可结合此思想
```

## 七、一句话总结

> **Mooncake是月之暗面(Kimi)的生产级LLM服务架构，以KV Cache为中心将Prefill和Decode分离到不同GPU集群，利用CPU DRAM/SSD构建分布式缓存池，配合缓存感知调度和预测式拒绝策略，在满足延迟SLO的前提下实现最高525%的吞吐量提升，代表了LLM服务系统从单机优化到集群架构优化的范式转变。**

---

*解读日期：2026-03-25*
