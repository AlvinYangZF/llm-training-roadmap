# Online Scheduling for LLM Inference 论文深度解读

**论文:** Online Scheduling for LLM Inference with KV Cache Constraints
**作者:** Patrick Jaillet (MIT), Jiashuo Jiang (HKUST), Konstantina Mellou (Microsoft Research), Marco Molinaro (Microsoft Research), Chara Podimata (MIT Sloan), Zijie Zhou (MIT / Microsoft Research)
**arXiv:** 2502.07115
**代码:** 无（理论性论文）

---

## 一、解决什么问题？

现有LLM推理调度研究主要从**工程视角**出发，缺乏严格的**理论保证**。本文首次从在线优化和运筹学角度，对LLM推理调度问题建立形式化模型并证明算法的理论性质。

```
LLM推理调度的三大独特挑战（区别于经典调度问题）:

挑战1：顺序生成约束
  token t+1 只能在 token t 生成后才能开始
  → 一个请求的所有token必须连续处理（非抢占式）
  → 不同于普通批处理任务可以中断重排

挑战2：KV Cache 内存动态增长
  处理请求i的第j个token时，内存占用 = s_i + j
  （s_i = prompt大小，随每个token增加1）
  请求i完整处理完后占用峰值 = s_i + o_i
  请求完成后内存清零
  → 内存约束是动态的，不同于静态内存需求

挑战3：毫秒级实时决策
  LLM推理规模下无法运行整数规划
  必须在极短时间内做出调度决策
  → 多项式时间算法是必须条件

对比经典调度问题:
  经典批处理调度: 任务内存需求固定，随时间不变
  LLM调度:       内存需求随时间线性增长，直到请求完成
  → 现有调度算法直接应用会失效（经典竞争比分析不适用）

ChatGPT的实际运营成本约$70万/天；合理调度可节约能源、降低成本。
```

**核心问题：LLM推理调度的理论下界是什么？能设计出具有可证明竞争比的多项式时间在线算法吗？**

---

## 二、核心方法/关键发现

### 形式化模型

```
模型设定:
  单个GPU Worker，内存容量 M
  请求集合 I = {请求1, 2, ..., n}
  请求 i 的参数: (a_i, s_i, o_i)
    a_i = 到达时间
    s_i = prompt长度（token数）
    o_i = 输出长度（token数，到达时未知）

请求处理的两个阶段:
  Prompt阶段: 处理prompt，生成第一个输出token
    内存需求 = s_i
  Token阶段:  逐步生成输出token（共o_i步）
    第j步内存需求 = s_i + j
    峰值（最后一个token）= s_i + o_i

非抢占约束:
  一旦请求i加入batch，必须连续运行o_i轮直到完成
  不能中途暂停（避免KV cache丢失和重计算开销）

内存可行性约束（任意时刻t）:
  sum_{i in S^(t)} (s_i + o_i^(t)) <= M
  其中 S^(t) = 时刻t的活跃请求集合
       o_i^(t) = 请求i在时刻t已生成的token数

优化目标: 最小化总端到端延迟（TEL）
  TEL(I; A) = sum_{i in [n]} (c_i(A) - a_i)
  c_i(A) = 算法A处理请求i最后一个token的时刻
```

### 关键理论发现1：下界不可能定理

```
定理4.1（不可能结果）:
  任何确定性在线算法的竞争比至少为 Omega(sqrt(n))

直觉解释:
  当到达过程是对抗性的（adversarial），
  在线算法无法获知未来请求的o_i（输出长度）
  → 可能做出导致内存死锁的批次决策
  → 在最坏情况下，性能差距随请求数增长

  与经典scheduling的对比:
  经典job scheduling: 可以达到 O(1) 竞争比
  LLM scheduling:     Omega(sqrt(n)) 竞争比（由KV cache的记忆性导致）
  → KV cache使LLM调度从理论上变得更难！
```

### 关键理论发现2：MC-SF算法的常数竞争比

```
尽管最坏情况无法避免，MC-SF算法在结构化假设下达到 O(1) 竞争比：

定理4.3（MC-SF的竞争比）:
  考虑所有请求在t=0同时到达、prompt大小相同（s_i = s）的实例
  假设预测输出长度满足 o_i <= tilde_o_i <= alpha * o_i
  若 M >= 2 * max_i(s_i + tilde_o_i)（内存至少是单请求峰值的两倍）
  则 MC-SF 是 O(1)-competitive 的

关键洞察:
  MC-SF通过预测输出长度 tilde_o_i（o_i的上界估计）
  主动规避内存溢出风险
  → 在输出长度有合理预测的情况下，可以突破最坏情况下界
```

### MC-SF 算法详解

```
MC-SF = Memory Constrained Shortest First（内存约束最短优先）

每轮 t 的处理逻辑:

  S^(t) = 已启动但未完成的请求集合
  R^(t) = 尚未开始的等待请求集合

  Step1: 优先处理已在途请求（S^(t)）
  Step2: 从等待队列选择新请求加入本轮batch：

    for 请求i in R^(t)（按预测输出长度 tilde_o_i 升序排列）:
      设 t_max(U) = t + max_{i in U} tilde_o_i（预测最晚完成时刻）
      检查可行性：对所有 t' in [t+1, t_max(U)]:
        sum_{i in S^(t)} (s_i + t' - p_i) * 1{tilde_o_i >= t'-p_i}
        + sum_{i in U} (s_i + t' - t) * 1{tilde_o_i >= t'-t} <= M
      若可行 → 将i加入 U^(t)，否则停止

    批次 = S^(t) 并 U^(t) 一起处理

关键设计决策:
  1. 优先处理已在途请求（减少每个请求的等待时间）
  2. 按预测输出长度升序选新请求（Shortest First策略）
     → 优先处理短请求，减少对内存的持续占用
     → 峰值内存主要由 s_i + tilde_o_i 决定，短请求峰值低
  3. 只在预测完成时刻检查可行性（而非所有时刻）
     原因：内存需求在完成时刻达到峰值，中间线性增长
           满足peak points等价于满足整个时间段

算法复杂度: O(M^2) per round（与请求数无关！）
```

---

## 三、技术细节

### 竞争比证明（Theorem 4.3 证明思路）

```
证明框架：分组求和法

将输出长度 1, ..., o_max 按2的幂次分组：
  U_ell = {输出长度 in [2^ell, 2^{ell+1}) 的请求集合}

关键引理4.5:
  MC-SF处理 U_ell 中第一个和最后一个请求的时间间隔:
  t_bar - t_underline <= 192/M * sum_{o=2^ell}^{o_bar_ell} n_o * vol_o + 5*o_bar_ell

  其中 vol_o = s*o + o*(o+1)/2 = 单个输出长度为o的请求的内存"体积"

引理4.4（MC-SF总延迟上界）:
  MC-SF <= 1536/M * sum_o n_o * sum_{o'<=o} n_{o'} * vol_{o'} + 24 * sum_o n_o * o

引理4.7（OPT总延迟下界）:
  OPT >= 1/(6M) * sum_o n_o * sum_{o'<=o} n_{o'} * vol_{o'} + 1/6 * sum_o n_o * o

结合两个界限:
  MC-SF <= O(1) * OPT （常数约 1536/M * 6M = 9216，但量级是O(1)）

直观理解:
  内存约束M迫使请求不能同时处理太多（分母中的M）
  但MC-SF的Shortest First策略恰好最小化了"内存体积的累积"
```

### Hindsight Optimal 整数规划基准

```
整数规划（IP）形式化最优调度:

min sum_{i in [n]} ( sum_{t=a_i}^{T} t * x_{i,t} + o_i - a_i )  ... (1)

s.t.
  sum_{t=a_i}^{T} x_{i,t} = 1,                             ∀i  ... (2)
  sum_{i=1}^{n} sum_{k=max{a_i, t-o_i}}^{t-1} (s_i+t-k)*x_{i,k} <= M,  ∀t  ... (3)
  x_{i,t} in {0,1}                                              ... (4)

变量解释:
  x_{i,t} = 1 表示请求i在时刻t开始处理

约束(2): 每个请求恰好调度一次
约束(3): 任意时刻t，内存总需求不超过M
          请求i若在时刻k开始，在时刻t的内存需求 = s_i + (t-k)
约束(4): 二元决策变量

局限: 此IP规模随n和T增长，只能在小规模实例上求解（用Gurobi）
     作为hindsight optimal基准，不用于实时调度
```

---

## 四、实验结果

### 合成数据实验

```
实验设置:
  内存容量 M 均匀随机 ~ [30, 50]
  prompt大小 s_i ~ 均匀 [1, 5]
  输出长度 o_i ~ 均匀 [1, M-s_i]
  200次独立实验

Arrival Model 1（全部同时到达，t=0）:
  消除信息不对称，仅测算法本身的次优性
  结果: MC-SF 平均延迟比 hindsight optimal 高 0.5%
        200次实验中有 114次完全等于最优解
  → MC-SF几乎就是最优算法！

Arrival Model 2（随机在线到达）:
  模拟真实在线不确定性
  结果: MC-SF 平均延迟比 hindsight optimal 高 4.7%
  → 即使在完全在线设定下，MC-SF接近最优
  → 4.7%的差距主要来自信息不对称（未来到达未知）
```

### 真实数据集实验

```
数据集:
  Vicuna/Chatbot Arena 对话数据集
  来源: 210,000+ 个独立IP地址
  规模: 数万个LLM推理请求

模型: Llama2-70B（在A100 GPU上模拟）

评估指标:
  总端到端延迟（TEL）
  与6种参数化基线算法对比（在高低负载两种设置下）

基线算法:
  FCFS（先到先服务）
  SJF（最短作业优先）
  各种基于输出长度预测的参数化变体

MC-SF结果（高负载设置）:
  MC-SF 显著超越所有6种基线配置
  延迟降低幅度在高负载下更为明显

MC-SF结果（低负载设置）:
  MC-SF仍显著优于所有基线
  → 在各种负载情况下均具有鲁棒性

实验意义:
  真实数据验证了理论保证之外的实际效果
  MC-SF的优势"超出了论文中放置的理论假设"
```

---

## 五、核心启示与局限

### 核心启示

```
1. LLM调度从理论上比经典调度更难
   KV cache的"记忆性"（内存需求随时间增长）导致
   最坏情况下竞争比为 Omega(sqrt(n))
   这是一个本质困难，不可通过聪明的算法规避
   → 工程优化需要配合输出长度预测才能在理论上可行

2. 输出长度预测是解锁好调度算法的钥匙
   MC-SF在有预测（tilde_o_i >= o_i）时达到O(1)竞争比
   而输出长度预测精度正在快速提升（文献报告可达80%）
   → 理论分析提供了清晰的"为什么预测重要"的量化依据

3. Shortest First（最短优先）在内存约束场景下是正确的
   MC-SF按预测输出长度排序选择新请求
   不是直觉上"公平"，而是理论上最优的内存利用策略
   → 短请求"体积"小（vol = s*o + o^2/2），优先处理减少整体延迟

4. Hindsight Optimal作为基准的价值
   IP表达的最优解为评估在线算法提供了严格基准
   (而非工程系统中常见的"与另一个启发式比较")
   → 实验发现MC-SF几乎没有"算法次优性损失"（0.5%）
      4.7%的gap来自信息不对称——这已是理论下界

5. 对实际运营的意义
   更好的调度 = 更少GPU浪费 = 更低成本 + 更少能耗
   ChatGPT日均运营成本$70万、能耗超50万度电
   即使节省几个百分点也有巨大的绝对价值
```

### 局限性

```
1. 单GPU Worker模型
   论文只分析单个GPU的调度
   多GPU（张量并行、流水线并行）下的调度未覆盖
   生产系统通常有数十到数千个Worker

2. 非抢占假设过强
   现实中vLLM等系统支持"抢占"（preemption）
   即将正在生成的请求挂起、KV cache卸载到CPU、腾出内存
   论文的非抢占假设使模型更简洁，但降低了实用性

3. 输出长度预测假设 tilde_o_i >= o_i
   MC-SF需要"overestimate"（预测值不小于真实值）
   低估输出长度会导致内存超限（违反约束(3)）
   真实预测方法更难保证单调上界性质

4. 竞争比常数过大
   定理4.3证明的O(1)中的常数约9216（1536/M * 6M）
   在实践中这个界极其宽松，实验中实际比值约1.005~1.047
   → 理论界与实验值的差距说明分析仍有改进空间

5. 模型简化
   离散时间步骤假设（batch processing time = 1）
   忽略prefill和token generation的延迟差异
   忽略了context caching（prefix reuse）的影响
```

### 在知识体系中的位置

```
LLM推理优化的两条路线:

工程路线（系统论文）:
  → vLLM（PagedAttention）
  → Orca（iteration-level scheduling）
  → Sarathi（chunked prefill）
  → FastServe（multi-level priority queue）
  优点：实践效果显著；缺点：缺乏理论保证

理论路线（本文的方向）:
  → 本文（MC-SF, 竞争比分析）
  → Bari et al. (2025)（类似模型，不同约束）
  → Li et al. (2025a)（throughput/stability模型）
  优点：可证明性质；缺点：模型可能过度简化

本文的独特价值:
  · 首次严格证明LLM调度的理论下界
  · 首次设计具有竞争比保证的多项式时间在线算法
  · 将运筹学/在线算法领域引入LLM系统研究
  · 为工程优化提供理论基础，指出"预测输出长度"的核心地位

未来方向:
  多Worker调度、抢占式调度、可变prompt大小（Wang et al. 2025）
  以及将理论模型与工程实现进一步接轨
```

---

*解读日期：2026-04-07*
