# TetriInfer 论文深度解读

**论文:** Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads
**作者:** Cunchen Hu, Heyang Huang, Liangliang Xu, Xusheng Chen, Jiang Xu, Shuang Chen, Hao Feng, Chenxi Wang, Sa Wang, Yungang Bao, Ninghui Sun, Yizhou Shan
**机构:** 中国科学院大学 (ICT, CAS)、华为云
**arXiv:** 2401.11181
**发表时间:** 2024年1月

---

## 一、解决什么问题？

不同下游任务的LLM推理请求在token长度上差异极大，混合运行时互相干扰严重：

```
实际工作负载的长度分布（差异超两个数量级）:

  任务类型    Prompt长度    生成长度
  ──────────────────────────────────
  对话         ~18 tokens   短
  摘要生成     ~2000 tokens  短
  内容创作     短            ~512 tokens

  → 同一集群同时处理多种任务
  → 不同长度请求的资源需求差异极大
  → 混合运行 = 严重干扰！
```

### 三种干扰的量化测量

```
干扰1: Prefill + Prefill
  7个并发轻量Prefill(18 token): 延迟 2×
  63个并发轻量Prefill: 延迟 8×
  1个重量Prefill共存: 延迟 10×
  原因: token总数超过加速器饱和阈值(ChunkSize=512)

干扰2: Prefill + Decode
  1个重量Prefill进入batch: Decode延迟 5×！
  7个轻量Decode共存: Prefill延迟增加
  原因: Prefill计算密集，Decode内存密集，共置竞争

干扰3: Decode + Decode
  重量Decode(>512 tokens)混入: 吞吐量 -16%，延迟 +23%
  原因: 不同生成长度导致内存带宽和容量争用
```

**核心问题：** 如何设计一个云规模LLM推理系统，消除混合工作负载下的各类干扰，同时降低TTFT、JCT和推理成本？

---

## 二、核心方法/关键发现

### TetriInfer 三大设计支柱

```
                  混合工作负载请求
                        │
          ┌─────────────▼─────────────┐
          │    集中式控制平面           │
          │ (Global Scheduler +        │
          │  Cluster Monitor)          │
          └──────┬──────────────┬──────┘
                 │              │
        ┌────────▼────────┐  ┌──▼────────────┐
        │  Prefill 实例   │  │  Decode 实例  │
        │                 │  │               │
        │ 调度器          │  │ 接收器        │
        │ 长度预测器 ─────┼─►│ 调度器        │
        │ 主LLM(Chunked)  │  │ 主LLM(Decode) │
        │ KV Cache ───────┼─►│ KV Cache      │
        └─────────────────┘  └───────────────┘

支柱1: Chunked Prefill (分块Prefill)
支柱2: Prefill/Decode 解耦
支柱3: 两级调度 + 长度预测
```

### 支柱1：Chunked Prefill

```
核心观察: 加速器吞吐量在 ChunkSize 处饱和（OPT-13B: 512 tokens）

传统Prefill:
  │████████████████████████│  (不定长，可能超出ChunkSize)
  → 过短: GPU未饱和，效率低
  → 过长: 延迟增加，干扰其他请求

Chunked Prefill:
  将所有请求切片并填充到 ChunkSize:
  R1[512]  R2[256+256]  R3[128+128+256]
  → 每个iteration恰好 ChunkSize tokens
  → 加速器始终以饱和状态运行
  → 消除Prefill内部干扰

效果: 降低平均Prefill延迟 86.4% (vs vanilla vLLM)
```

### 支柱2：Prefill/Decode 解耦

```
虚拟化解耦设计:
  - Prefill实例 和 Decode实例 是虚拟概念
  - 可以在相同硬件资源上动态翻转(flip)角色
  - Instance Flip: 5~7ms完成翻转（仅改变内部变量）

KV Cache传输:
  → 完成Prefill后将KV Cache发送至指定Decode实例
  → 网络栈分三类: Direct(NVLink), Direct-NIC, Indirect(TCP)
  → 传输粒度: 请求级(request-level)传输，减少传输次数
```

### 支柱3：两级调度 + 长度预测

```
调度问题: 如何将Decode请求分配给合适的Decode实例？
关键难点: 不知道请求会生成多少token！

解决方案: 长度预测模型 (Length Predictor)
  → 使用小型LLM (OPT-125M) 对目标LLM (OPT-13B) 的输出分类
  → 预测粒度: 200 tokens为一档 (0, 200, 400, ...)
  → 准确率: 74.9% (对于200-token粒度)
  → 运行位置: 并行模式下与主LLM同时运行（10%吞吐量损耗）

Prefill调度策略（三种）:
  FCFS: 先来先服务，简单，但长请求导致HOL阻塞
  SJF: 最短优先，降低平均JCT，可能饿死长请求
  LJF: 最长优先，优化TTFT for long prompts

  使用PrefillSchedBatch防止饥饿:
  批次大小设为16时，SJF降低平均等待时间 7.8% vs FCFS
```

---

## 三、技术细节

### Decode 实例调度

```
工作集感知(Working-Set Aware)调度策略:

传统vLLM贪心策略: 只要有空间就加入请求
  问题: 未来iteration可能内存不够 → swap/thrashing

Reserve-Static: 预测内存使用 < 可用内存 → 才加入
  → 静态保守

Reserve-Dynamic: 考虑最短剩余任务完成时的内存预测
  → 主动预防内存抖动
  → 最多降低平均JCT 10% vs vLLM贪心策略

Decode实例选择算法 (分配器):
  1. 将Decode实例分为两集: α(资源足够) 和 β(不足)
  2. 用"power-of-two"算法从α中随机选2个候选
  3. 选择重量Decode:轻量Decode比率最低的那个
  → 将重量Decode请求均匀分散，避免局部过热
```

### Instance Flip（实例翻转）

```
翻转场景: Prefill实例负载 < 10% → 翻转为Decode实例

Prefill → Decode 翻转步骤:
  S1: Global Scheduler 通知 Prefill 实例停止接收请求
  S2: Prefill 实例排空队列
  S3: 翻转完成（仅改内部状态变量）
  S4: Global Scheduler 确认新角色

翻转时间: 5~7 ms (不重启进程，不重新加载模型)
意义: 动态响应负载变化，减少资源浪费
```

### 系统架构

```
集中式控制平面:
  ├─ Cluster Monitor: 每100ms收集各实例负载信息
  │   定期广播Decode实例负载给所有Prefill实例
  └─ Global Scheduler: 维护请求状态表，路由到最轻负载Prefill实例

Prefill实例组件:
  ├─ 本地调度器 (FCFS/SJF/LJF)
  ├─ 长度预测器 (小型LLM，预测Decode长度)
  ├─ 主LLM引擎 (Chunked Prefill)
  └─ 分配器 (选择目标Decode实例)

Decode实例组件:
  ├─ 接收器 (接收KV Cache)
  ├─ 本地调度器 (工作集感知)
  └─ 主LLM引擎 (基于vLLM)
```

---

## 四、实验结果

### 测试配置

```
硬件: 4× NVIDIA V100 (32GB HBM), 单服务器
模型: OPT-13B (主LLM), OPT-125M (长度预测)
网络模拟: 200~300 Gbps (RoCE/NVLink)
对比基准: vanilla vLLM
数据集: ShareGPT (公开)
评估指标: TTFT, JCT (Job Completion Time), Perf/$
```

### 端到端性能对比

```
轻量Prefill + 轻量Decode (LPLD) — 对话场景:
  TTFT:   -44%  (两种硬件设置下均如此)
  JCT:    -40%
  资源使用: 约2× (但完成速度2×，因此Perf/$ 提升1.4×)

轻量Prefill + 重量Decode (LPHD) — 内容创作场景:
  TTFT:   -97%  ← 因为彻底消除了Prefill/Decode干扰
  JCT:    -47%
  Perf/$: 提升 2.4×

重量Prefill + 轻量Decode (HPLD) — 摘要+对话:
  TTFT:   -9%
  JCT:    -23%
  资源增加 43%，但Perf/$ 仍提升 14%（vLLM过于低效）

混合工作负载 (Mixed) — 实际场景:
  TTFT:   -85%
  JCT:    -50%
  资源:   -21%
  Perf/$: 提升 1.9×
```

```
综合结论：
  TetriInfer 使用 38% 更少资源
  同时降低平均TTFT 97%，平均JCT 47%
  Perf/$ 提升 2.4× (最佳情况)
```

---

## 五、核心启示与局限

### 核心启示

```
1. 干扰是云规模LLM推理系统的头号效率杀手
   → 不同任务类型的token长度差异超2个数量级
   → 简单地把所有请求扔进一个系统 = 严重效率损失

2. Chunked Prefill 是保持加速器高效利用的关键
   → 不足ChunkSize: GPU效率低
   → 超过ChunkSize: 干扰其他请求
   → 精确控制每次iteration = ChunkSize是最优策略

3. 长度预测使调度从"盲目"变为"知情"
   → 不需要精确预测，粒度为200的分类已足够
   → 74.9%的准确率已能显著减少热点
   → 小模型(125M)的推理开销远小于调度收益

4. 实例翻转让解耦不增加总成本
   → 动态调整Prefill/Decode实例比例
   → 负载低时翻转角色，不浪费资源
   → 这是TetriInfer vs DistServe的关键差异
```

### 局限性

```
1. 对重量Prefill + 重量Decode (HPHD) 效果有限
   → 两阶段都资源密集，解耦收益边际较小
   → 引入的Overhead无法完全被覆盖
   → 论文承认这是设计局限

2. 长度预测在极端情况下失准
   → LLM输出受temperature/top-p影响，同问题可能差异巨大
   → 预测粒度=200时降低部分精度换取可行性
   → 预测错误时Reserve-Static/Dynamic策略失效

3. 评估硬件较老（V100），与当前H100/A100有差距
   → ChunkSize等参数会随硬件变化
   → 实际部署效果需重新测试

4. 网络栈实现受限
   → 当前只实现了Indirect（TCP）传输类型
   → Direct-NIC和Direct的性能优势未在真实硬件上验证
```

---

## 六、与 DistServe/Splitwise 的比较

```
三篇论文的差异定位:

           DistServe      Splitwise      TetriInfer
  关注点    goodput最大化  硬件异构效率   混合负载干扰
  调度      并行配置优化   池化+路由      长度预测调度
  翻转      静态分配       Mixed池缓冲    动态Instance Flip
  硬件      同构GPU        异构A100+H100  同构GPU
  生产traces 无             Azure traces  无（模拟）
  特色创新  自动放置算法   层级KV传输     Chunked Prefill

共同贡献: 均独立发现并验证了Prefill/Decode分离的价值
```

---

## 七、一句话总结

> **TetriInfer通过实验量化了混合LLM推理工作负载中三类干扰（Prefill间、Prefill-Decode间、Decode间），提出Chunked Prefill保持加速器满负荷、Prefill/Decode解耦消除阶段间干扰、以及基于小LLM长度预测的两级调度消除Decode内干扰三大支柱，使用38%更少资源同时将平均TTFT降低97%、平均JCT降低47%，并实现1.9×的性能成本比提升。**

---

*解读日期：2026-04-07*
