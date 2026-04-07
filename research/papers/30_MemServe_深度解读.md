# MemServe 论文深度解读

**论文:** MemServe: Flexible Mem Pool for Building Disaggregated LLM Serving with Caching
**作者:** Cunchen Hu, Heyang Huang, Junhao Hu, Jiang Xu, Xusheng Chen, Tao Xie, Chenxi Wang, Sa Wang, Yungang Bao, Ninghui Sun, Yizhou Shan (华为云 / 中科院计算所 / 北京大学)
**arXiv:** 2406.17565
**代码:** 未公开

---

## 一、解决什么问题？

现代LLM服务系统正在演化为有状态系统，核心在于KV cache的复用。现有优化技术分为两类：

```
跨请求优化（Inter-request）:
  代表：Context Caching（SGLang RadixAttention、Pensieve等）
  原理：相同前缀的请求共享KV cache，避免重复预填充
  局限：设计时未考虑Prefill-Decode分离场景

单请求内优化（Intra-request）:
  代表：Disaggregated Inference（DistServe/Splitwise）
         Sequence Parallelism（InfiniteLLM）
  原理：将单请求拆分到多实例并行处理
  局限：各实例间缺乏KV cache传递和复用的统一机制
```

**两类优化无法共存的根本矛盾：**

```
问题1：现有系统无法同时使用 Inter-request + Intra-request 优化
  Disaggregated Inference 将请求拆成子请求
  → 破坏了 context caching 对完整KV cache的依赖
  → Decode实例无法将 KV cache 回传给 Prefill 实例复用

问题2：没有统一的KV cache管理层
  各系统自己管内存，数据格式、寻址方式、网络传输原语各不相同
  → 跨实例KV cache传输效率低（离散内存布局 + NIC/DRAM额外拷贝）
  → 无法在系统层面最大化KV cache复用率

核心问题：如何构建一个统一的内存抽象层，让各类优化技术
          能在同一系统中无缝组合？
```

---

## 二、核心方法/关键发现

### 核心架构：三层组件

```
MemServe 系统 = MemPool（弹性内存池）
              + Disaggregated Inference with Caching
              + Locality-Aware Global Scheduler

     Global Request Scheduler
     ┌──────────────────────────────────┐
     │  Global Prompt Tree (分布式前缀树) │
     │  Locality-Aware Scheduling Policy │
     └──────────────────────────────────┘
              │路由请求
     ┌─────────────────────────────────┐
     │  Prefill-Only │ Decode-Only │ PD-Colocated │
     │  Instance     │ Instance    │ Instance     │
     │  MemPool ─────┼─────────────┼──── MemPool  │
     │  (GPU HBM     │             │    DRAM)      │
     └─────────────────────────────────┘
              │ NVLink/PCIe/RoCE/HCCL/IB
```

### 关键发现：MemPool的三类API设计

```
API类别1：内存块操作（Memory Block）
  alloc_mem(size, type, id)    # 在指定实例上分配HBM或DRAM
  free_mem(addrList)           # 释放内存
  swap_out(num_blocks)         # HBM → DRAM
  swap_in(addrList)            # DRAM → HBM

API类别2：索引操作（Index）
  insert(tokenList, addrList)  # 建立 prompt token → KV cache 映射
  match(tokenList)             # 查找已缓存的KV cache地址
  delete(tokenList)            # 删除缓存项

API类别3：分布式传输（Distributed Transfer）
  transfer(id, src, dst, flags)            # 跨实例传输KV cache
  transfer_with_insert(id, tokenList, ...) # 传输并同步建立索引
                                           # 省去一次网络往返
```

### 关键发现：disaggregated inference + caching 的渐进设计路径

```
阶段1 PD-Basic（基础disaggregated）:
  Prefill实例计算后 → transfer → Decode实例
  问题：无caching，多轮对话重复传输相同KV

阶段2 PD-Caching-1（Prefill侧缓存）:
  Prefill完成后 insert → 保存历史KV
  下次相同前缀 → match命中 → 节省Prefill计算
  问题：Decode实例不知道历史KV，多轮还是要重传

阶段3 PD-Caching-2（Decode侧缓存 + 减少数据移动）:
  Prefill改用 transfer_with_insert
  → Decode实例接收时同步建立自己的本地索引
  → 多轮对话：Prefill只需增量传输新KV，历史KV Decode已有
  问题：Prefill实例没有Decode阶段产生的历史KV

阶段4 PD-Caching-3（全面缓存，首次在disaggregated架构实现）:
  请求完成时，Decode实例调用 transfer_with_insert
  → 将Decode产生的KV回传给Prefill实例
  → Prefill实例历史KV持续增长
  → 多轮对话的caching收益随轮次线性增长！
```

---

## 三、技术细节

### 内存布局与网络传输优化

```
挑战：KVCache是分散的小块内存（每层2个block），
      NCCL等集合通信原语每次传输需独立API调用
      → 对于2*L个离散block，需要 2*L 次网络API调用

优化：块聚合（Block Aggregation）
  将若干小KV block聚合成一个大页（类似hugepage）
  原来2个block/层 → 聚合为1个大block/层
  API调用次数: 2*L → 1 (by-request-agg模式)

传输方式对比:
  By-Layer:      每层处理完立刻传输 (延迟低但API调用多)
  By-Request:    整个请求完成后一次性传输 (API调用少但延迟高)
  By-Req-Agg:    By-Request + 聚合大块 (兼顾两者优势)
  → 高负载下 by-req-agg 胜出，低负载 by-layer 更优
```

### 代价模型

```
exec(x, y) = 预测以缓存比例y处理长度x的prompt的执行时间

三类算子分别建模:
  (a) 计算密集型（Compute-bound）:
      op(x, y) = (η-1)*T_fullwave + T_lastwave
      其中 η = ceil(B_total/SMsnum)

  (b) 内存密集型（Memory-bound）:
      op_attention(x, y) = ax^2*y + bx^2 + cx + d
      通过 profiling 拟合系数

  (c) 常数型（Constant）: 归一化、激活等
      线性关系直接拟合

Operator级代价模型优于Architecture级:
  TP=2 时, arch-level模型精度下降20%（Amdahl定律导致）
  op-level模型可简单乘以常数因子处理TP/PP变化
```

### 全局调度器的局部性感知策略

```
Global Prompt Tree（全局前缀树）:
  维护三类实例（Prefill-Only、Decode-Only、PD-Colocated）的
  分布式前缀树，每个节点记录哪个实例持有对应KV cache

调度流程:
  1. 请求到达 → tokenizer
  2. 并行查询所有类型实例的前缀树 match
  3. 返回每个实例的缓存比例 y_p
  4. Policy Module 选择 longest common prefix 的实例
  5. 检查是否有其他实例持有额外历史KV（可转移）
  6. 决定：直接recompute 还是 transfer额外的KV
     条件：transfer(y_p, y_p') <= exec(x, y_p) - exec(x, y_p')

调度策略对比（LooGLE数据集，80个session，250个请求，Share ratio=3）:
  Least Load（无局部性感知）     P99 TTFT 基准
  Session-ID-Based（会话级）     改善 intra-session caching
  Prompt-Tree-Based（本文）      P99 TTFT 提升 59%（最优）
  → 前缀树策略跨会话复用，caching机会最多
```

---

## 四、实验结果

### 实验环境

```
硬件: 单NVIDIA DGX H800服务器
GPU:  8块H800-80GB，NVLink 400GB/s
CPU:  192核 Intel Xeon Platinum 2.4GHz
内存: 2 TB DRAM
软件: Ubuntu 20.04, CUDA 12.2
模型: Llama2-13B (TP=2)，4个推理实例（1P1D配置）
数据集: ShareGPT、LooGLE（长文档QA）、ReAct（Agent推理）
```

### ShareGPT 数据集结果

```
1P1D vs PD（无缓存）:
  平均JCT: -30%（减少），P99 JCT: -42%
  disaggregated inference 显著改善 TPOT

1P1D-CC vs 1P1D（加缓存）:
  平均TTFT: -58%，P99 TTFT: -45%（缓存复用减少计算）
  平均JCT:  -17%，P99 JCT:  -29%
```

### LooGLE 数据集结果（长文档QA，大量共享前缀）

```
Disaggregated Inference 提升（vs PD-colocated）:
  平均JCT: -10.3%，P99 JCT: -10.8%

加入 Context Caching:
  平均JCT: -26.9%，P99 JCT: -22.5%
  平均TTFT: -56.2%，P99 TTFT: -45.2%
→ 长文档场景缓存收益显著（高前缀共享率）
```

### ReAct 数据集结果（Agent推理，极长序列）

```
Disaggregated Inference:
  平均JCT: -40.8%，P99 JCT: -53.1%

加入 Context Caching:
  平均JCT额外降低 26.7%，P99 JCT 额外降低 21.4%
  平均TTFT额外降低 78.5%，P99 TTFT额外降低 84.9%
→ Agent工作负载收益最大（轮次多，前缀重用率高）
```

### 微基准测试

```
MemPool API 延迟:
  内存操作: 约800ns/block（轻量）
  insert/match（4K token）: < 0.7ms

块聚合效果（Block Aggregation）:
  原始离散布局 vs 聚合块：传输时间提升一个数量级以上
  单NCCL communicator足够（块足够大时）

MemPool Caching Study:
  vLLM原版hash prefix机制: prompt增长时开销急剧上升
  MemPool Radix Tree: 大prompt时开销最小（几乎平坦）
```

---

## 五、核心启示与局限

### 核心启示

```
1. "统一内存抽象"是突破系统优化孤岛的关键
   MemPool作为底层基础设施，让各种优化技术
   （disaggregated、caching、sequence parallel）
   能组合使用，而不是互相排斥

2. 渐进式设计路径具有实用价值
   PD-Basic → PD-Caching-1 → 2 → 3 的四步演进
   清晰展示了如何逐步引入caching到disaggregated架构
   每步有明确的trade-off分析

3. 跨请求缓存的关键是调度，不只是存储
   Global Prompt Tree使跨会话、跨实例的KV cache
   复用成为可能，比Session-ID策略多复用59%的TTFT

4. 网络原语与AI负载的不匹配是系统设计盲区
   NCCL为集合通信设计，不适合P2P小块传输
   → 块聚合 + 自定义传输策略是务实解法
```

### 局限性

```
1. 单机评估
   所有实验在单台8-GPU DGX服务器上完成
   跨机网络（RDMA/InfiniBand）场景的性能未验证

2. 模型规模受限
   仅评估 Llama2-13B（TP=2，4个实例）
   超大模型（70B+）或MoE模型行为未知

3. 内存层次管理不完整
   RDMA-based传输尚未实现（仅NCCL send/recv）
   DRAM ↔ HBM 的swap机制在多机场景尚待验证

4. 替换策略简单
   KV cache淘汰仍依赖LRU，未探索更智能的
   基于访问预测的替换策略

5. 代价模型精度
   Arch-level模型在TP=2时误差达20%；
   Operator-level更好但需要per-operator profiling
```

### 在知识体系中的位置

```
LLM Serving 优化技术栈:

  层面1 单请求KV管理      → PagedAttention (消除碎片)
  层面2 跨请求KV复用      → SGLang/Pensieve (context caching)
  层面3 阶段分离          → DistServe/Splitwise (P/D disaggregation)
  层面4 统一内存抽象      → MemServe（本文）— 首次整合3+4
  层面5 集群级KV池化      → Mooncake (全局KV cache池)
  层面6 超长上下文支持    → Infinite-LLM (分布式Attention)

  MemServe填补了"阶段分离"与"内容复用"之间的空白
  是首个同时支持disaggregated inference + context caching的系统
```

---

*解读日期：2026-04-07*
