# CachedAttention 论文深度解读

**论文:** Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention
**作者:** Bin Gao, Zhuomin He, Puru Sharma, Qingxuan Kang, Djordje Jevdjic, Junbo Deng, Xingkun Yang, Zhou Yu, Pengfei Zuo
**机构:** National University of Singapore, Shanghai Jiaotong University, Huawei Cloud
**arXiv:** 2403.19708 (2024年6月更新)

---

## 一、解决什么问题？

多轮对话是LLM服务最基本的功能之一，但现有推理系统对多轮对话极其低效：

```
多轮对话的重复计算问题:

  第1轮: 用户输入q1, 模型回复a1
         → 系统计算KV(q1,a1), 然后丢弃

  第2轮: 用户输入q2
         → 系统必须重新计算KV(q1,a1) + 计算KV(q2)
         → KV(q1,a1) 被重新计算了！

  第3轮: 用户输入q3
         → 重新计算KV(q1,a1,q2,a2) + 计算KV(q3)
         → 历史token的KV被计算了3次！

真实数据 (ShareGPT分析):
  → 73%的对话是多轮的
  → 30%的对话超过4K token
  → 到第20轮时: 99%的prefill计算用于历史token!
  → GPU时间占用: 历史prefill占比趋近100%

财务影响 (LLaMA-65B, 4×A100):
  prefill 2K tokens = 360ms = 5GB KV cache
  每14秒KV cache就填满GPU HBM
  → 无法保存 → 必须丢弃 → 下轮重算
```

问题的根源：**GPU HBM容量有限，多轮对话的KV cache远超HBM容量，系统被迫丢弃并反复重算**

**CachedAttention的目标：建立分层KV cache系统(HBM + DRAM + 磁盘)，在会话间隙保存KV cache，会话重激活时直接复用，消除历史token的重复计算。**

---

## 二、核心方法/关键发现

### 核心设计：AttentionStore + CachedAttention机制

```
传统Attention流程:
  会话非活跃 → 丢弃KV cache
  会话重新激活 → 从头prefill所有历史token

CachedAttention流程:
  会话非活跃 → 保存KV cache到AttentionStore
  会话重新激活 → 从AttentionStore加载历史KV
               → 只prefill新的input token
               → 节省99%的历史prefill计算

AttentionStore架构:
  ┌─────────────────────────────────────────┐
  │  CachedAttention Controller             │
  │  ┌──────────┐  ┌─────────────────────┐ │
  │  │    Job   │  │ AttentionStore Mgr  │ │
  │  │Scheduler │  │ KV Access (§3.2)   │ │
  │  │          │  │ KV Placement (§3.3) │ │
  │  └──────────┘  └─────────────────────┘ │
  └──────────────┬──────────────────────────┘
                 ↓
  ┌──────────────────────────────────────────┐
  │ GPU Cluster: HBM (高速但容量小)           │
  │ Host Memory: DRAM (中速中容量)           │
  │ AttentionStore: Disk/SSD (慢速但大容量)  │
  └──────────────────────────────────────────┘
```

### 四大技术挑战及解决方案

```
挑战1: KV cache访问开销高
  问题: 从DRAM加载KV到HBM的时间不可忽视
        例: 2K tokens (5GB) 从DRAM加载到GPU ≈ 192ms
        而重新prefill 2K tokens ≈ 360ms
        节省效果大打折扣

  解法: 层级预加载 (Layer-wise Pre-loading)
  → 利用Transformer逐层计算的特性
  → GPU计算Layer i时，同时从DRAM加载Layer i+1的KV
  → 计算与IO完全重叠，消除加载等待

挑战2: KV cache存储容量需求大
  问题: 磁盘(TB级) > DRAM(百GB) > HBM(GB级)
        大多数KV cache需要在磁盘中保存
        磁盘随机访问性能差

  解法: 调度器感知预取 (Scheduler-aware Fetching)
  → 利用作业调度器的未来知识
  → 预先将即将执行会话的KV cache从磁盘取到DRAM
  → 确保执行时KV cache已在DRAM中

挑战3: 上下文窗口溢出使保存的KV失效
  问题: 历史token超出上下文窗口 → 需要截断
        截断改变了每个token的位置 → 位置编码改变
        → 保存的KV cache中内嵌了旧位置编码 → 失效！

  解法: 解耦位置编码 (Decoupled KV Cache Truncation)
  → 保存KV时不嵌入位置编码 (使用RPE)
  → 截断时直接对KV cache做截断，无需重新计算
  → 加载KV时动态注入新位置编码

挑战4: 会话非活跃期间保存KV的额外开销
  问题: 保存KV cache到DRAM/磁盘需要时间
        这个保存时间在关键路径上，延迟后续请求

  解法: 异步保存 (Asynchronous Saving)
  → 在prefill+decode过程中，异步将KV cache写入AttentionStore
  → 保存操作与推理计算并行执行
  → 保存开销完全隐藏
```

---

## 三、技术细节

### 层级预加载机制

```
执行流程 (3层模型为例):

                prefill        decode
Execution:  Layer1 Layer2 Layer3  L1 L2 L3 L1 L2 L3
Read:           Layer1 Layer2 Layer3

关键:
  GPU执行Layer 1时 → 同时预取Layer 2的KV cache
  GPU执行Layer 2时 → 同时预取Layer 3的KV cache
  → 加载时间完全被计算时间覆盖

当KV cache加载时间 > 单层计算时间时:
  引入更大的预取缓冲区 (Pre-loading Buffer)
  缓冲区大小 = B × (T_load × L_hist - T_pref × L_new)
  → 缓冲区吸收时序不对齐的开销

预加载的效果 (LLaMA-13B, 1K历史token + 100新token):
  无预加载:   prefill时间 = KV加载时间 + 计算时间
  有预加载:   prefill时间 ≈ 计算时间 (降低35%~61%)
```

### 异步KV Cache保存

```
Prefill阶段:
  → 逐层生成KV cache
  → 每层KV cache生成后立即异步写入AttentionStore
  → 写入与后续层的计算并行

Decode阶段:
  → 每步生成一个新token的KV
  → 新KV异步写入，不阻塞decode

效果:
  执行流: [Layer1][Layer2][Layer3][L1][L2][L3]...
  写流:       [W1][W2][W3][W1'][W2'][W3']...
  → 两个流完全并行
  → 端到端推理执行时间减少13%~15%
```

### 调度器感知的取/换策略

```
预取策略 (Disk → DRAM):
  维护预取窗口 L_pw = C_mem / S_kv  (内存容量 / 平均KV大小)
  Job Scheduler有完整的等待队列信息
  → 扫描等待队列中的L_pw个job
  → 命中磁盘的job → 提前将其KV从磁盘取到DRAM

淘汰策略 (DRAM → Disk):
  维护淘汰窗口 (C_mem + C_disk) / S_kv
  → 在窗口内的job → 豁免淘汰 (即将用到)
  → 窗口外的job按优先级淘汰
  → 优先淘汰等待队列中最靠后的job (最不紧迫的)

关键优势: 利用调度器的"未来知识"
  LRU/FIFO: 只看历史 → 可能淘汰即将使用的KV
  CachedAttention: 看未来 → 不淘汰即将使用的KV
  → 缓存命中率提升27%~31%
```

### 位置编码解耦

```
问题根源:
  绝对位置编码(APE): token的KV中直接嵌入了位置信息
  → 截断后位置改变 → 内嵌位置信息错误 → KV失效

  相对位置编码(RPE: RoPE/ALiBi): 位置信息在Query和Key的运算时动态计算
  → KV本身不含位置信息 → 可以截断后重新计算位置

CachedAttention的做法:
  保存阶段:
    → 在RoPE编码加入之前保存KV (存纯Key向量)
    → AttentionStore中的KV是"无位置信息的"

  加载阶段:
    → 加载KV后动态注入新的位置编码
    → 截断操作直接对KV cache做切片，无需重算

KV Cache截断示意:
  原始KV: pos[0:2048]
  截断后: pos[0:1536]  (截去后512个token)
  CachedAttention: 直接截取KV[0:1536] + 重新embed位置
  传统方法: 必须重新计算KV[0:1536] (这正是瓶颈!)
```

---

## 四、实验结果

### 端到端性能提升

```
比较基准: RE (Recomputation) = 每轮重新计算所有历史KV

模型           CA vs RE: TTFT降低   Prefill提升   GPU时间减少
─────────────────────────────────────────────────────────
LLaMA-13B      85%↓              6.8×↑          4.0×↓
LLaMA-65B      61%↓              2.6×↑          1.9×↓
LLaMA-70B      87%↓              7.8×↑          3.3×↓
Falcon-40B     86%↓              7.2×↑          3.4×↓

TTFT = Time To First Token (首token延迟，用户感知最重要的指标)
→ CachedAttention将TTFT降低了61%~87%!
```

### 缓存命中率

```
实际部署测试 (9K个ShareGPT会话):

模型          缓存命中率 (DRAM+Disk)
──────────────────────────────────
LLaMA-13B         86%
LLaMA-65B         71%  (模型大，每token更大，缓存更少会话)
LLaMA-70B         89%
Falcon-40B        90%

说明:
  LLaMA-65B命中率较低因为每个token KV cache = 2.5MB
  相同存储空间只能容纳更少的会话
```

### 推理成本节约

```
基于AWS EC2定价 ($5/小时/A100, $0.0088/GB/小时 DRAM, $0.00082/GB/小时 SSD):

模型           CA vs RE 成本节省
────────────────────────────────
LLaMA-13B         70%
LLaMA-65B         43%
LLaMA-70B         66%
Falcon-40B        68%

存储成本占比 (LLaMA-70B):
  DRAM: 9.0% of total
  SSD:  9.0% of total
  → 仅18%的存储成本换来66%的总成本节约!
```

### 不同存储配置的影响

```
LLaMA-13B, 128G DRAM + 各种SSD配置:

           HBM-only  HBM+DRAM  CachedAttention (HBM+DRAM+10TB SSD)
命中率         ~0%      3-20%          86%
GPU时间        1×       1.3×           4×+

→ 单纯用HBM基本无效 (容量太小)
→ DRAM有帮助但容量仍不足
→ 加入SSD后命中率从20%跳到86%
```

### 调度器感知策略的价值

```
缓存命中率对比 (128G DRAM + 2TB SSD):

             CA命中率   LRU命中率   FIFO命中率
LLaMA-13B     86%        59%         55%
(128G/10T)

CA比LRU高27%, 比FIFO高31%
GPU时间: CA比LRU快2.7×
```

### 层级预加载的效果

```
LLaMA-13B, historical=1000 tokens, new=100 tokens:

方法                     Prefill时间
────────────────────────────────────
无预加载 (NO-PL)          100% (基准)
层级预加载 无buffer (PL-B0)  65% (降低35%)
层级预加载 5层buffer (PL-B5)  39% (降低61%)

→ 最大可消除61%的KV加载延迟
```

---

## 五、核心启示与局限

### 核心启示

```
1. 多轮对话是LLM服务的主要成本来源:
   73%的对话多轮 → 99%的prefill是冗余历史重算
   这是一个被系统界长期忽视的重要场景

2. KV cache是"有状态的资产"，值得跨轮次保存:
   传统视角: KV = 临时中间结果，用完即弃
   CachedAttention视角: KV = 会话状态，应跨轮复用
   → 存储成本(18%)远低于计算节省(66%)

3. 存储层次越丰富，系统越高效:
   只有HBM: 几乎无效 (容量太小)
   HBM+DRAM: 有帮助但不够
   HBM+DRAM+SSD: 86%命中率 → 大幅节省
   → 利用廉价大容量存储扩展有效KV容量

4. 调度器掌握"未来信息"，应充分利用:
   LRU/FIFO只看历史 → 可能淘汰即将使用的KV
   调度器知道哪些会话即将运行 → 保护对应KV
   → 命中率从~55%提升到~86%

5. 位置编码解耦是实用化的关键:
   没有解耦位置编码，上下文溢出时所有保存的KV失效
   ShareGPT中47%的会话会发生上下文溢出
   没有这个设计，CachedAttention的命中率会降低17-41%
```

### 局限性

```
1. LLaMA-65B命中率较低 (71%):
   每token 2.5MB KV → 相同存储容纳会话少
   随着模型更大，每token KV也更大 → 命中率下降
   可以通过量化KV来缓解

2. 到达率(session arrival rate)的影响:
   高到达率 → 更多并发会话 → 更多存储需求
   存储相同时，高到达率下命中率下降
   (从0.5/s到2.0/s: 命中率从82%降到77%)

3. 磁盘随机访问对性能有影响:
   尽管有预取，当预取不完美时磁盘延迟仍然显现
   磁盘带宽低于DRAM一个数量级

4. 仅针对多轮对话场景:
   单轮推理或不重复会话的场景无法获益
   对于无状态API使用模式帮助有限

5. 与RadixAttention的关系:
   CachedAttention处理"跨时间轴"的KV复用 (同session)
   RadixAttention处理"跨请求"的KV复用 (同前缀)
   两者互补但当前论文未展示深度集成
```

---

## 六、在知识体系中的位置

```
KV Cache利用模式的完整谱系:

  单请求内优化 (减少KV大小):
    H2O, Quest, InfiniGen → 减少需要计算/加载的KV数量

  跨请求共享 (同前缀复用):
    RadixAttention (SGLang) → 不同请求共享相同前缀的KV

  跨轮次保存 (多轮对话复用):
    CachedAttention → 同一会话的历史KV保存并复用
    解决了多轮对话中99%历史重算的问题

  跨设备存储扩展:
    FlexGen, InfiniGen → GPU offload到CPU
    CachedAttention → HBM + DRAM + Disk三级层次
    KVSwap → 设备端 CPU + Disk

CachedAttention的独特贡献:
  → 首次系统性解决多轮对话的KV重算问题
  → 三级存储层次 + 调度器感知策略
  → 位置编码解耦支持上下文窗口溢出
  → 在Huawei Cloud生产环境验证，实现70%成本节省
```

## 一句话总结

> **CachedAttention发现多轮对话中99%的prefill时间用于重新计算历史KV，提出三级分层KV缓存系统(HBM+DRAM+SSD)配合调度器感知的预取/淘汰策略，利用层级预加载和异步保存隐藏IO开销，以及位置编码解耦支持上下文窗口截断，实现了86%的缓存命中率、87%的TTFT降低、7.8×的prefill吞吐量提升和70%的端到端推理成本节省。**

---

*解读日期：2026-04-07*
