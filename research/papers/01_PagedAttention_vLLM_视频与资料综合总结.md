# PagedAttention & vLLM 综合总结

**基于以下资料整理：**
- Woosuk Kwon MLSys Seminar 演讲 (YouTube: Oq2SN7uutbQ)
- Woosuk Kwon CMU 2025 课程讲义
- vLLM 官方博客 (blog.vllm.ai)
- "Paged Attention from First Principles" by Hamza ElShafie
- DEV Community 技术解析文章
- SOSP 2023 原论文

---

## 第一部分：问题定义 — 为什么推理是内存瓶颈？

### 训练 vs 推理的本质区别

| 维度 | 训练 | 推理 |
|-----|------|------|
| 瓶颈类型 | 计算瓶颈（Compute-bound） | 内存瓶颈（Memory-bound） |
| 并行性 | 整个序列并行处理 | 逐token顺序生成 |
| 工作模式 | 矩阵-矩阵乘法（高GPU利用率） | 矩阵-向量乘法（低GPU利用率） |

### 推理的两个阶段

**1. Prefill（预填充）阶段**
- 输入：用户的完整prompt (x₁, ..., xₙ)
- 操作：并行计算所有prompt token的KV cache
- 特点：计算密集，GPU利用率高
- 输出：第一个生成token + 所有prompt token的KV缓存

**2. Decode（解码）阶段**
- 输入：每次一个新token
- 操作：利用已缓存的KV + 新token的KV，计算下一个token
- 特点：顺序执行，内存密集，GPU利用率低
- 结束：遇到 `<eos>` 或达到最大长度

### KV Cache 的内存开销

KV Cache 大小计算公式：
```
KV Cache = 2 × bytes_per_param × num_layers × batch_size × num_heads × head_dim × sequence_length
```

**实际案例：**

| 模型 | 单序列 KV Cache（最大上下文） | 说明 |
|-----|--------------------------|------|
| OPT-13B (2048 tokens) | 1.6 GB | 单个请求就占1.6GB |
| LLaMA-13B (4096 tokens) | 1.7 GB | 更长上下文，更大开销 |
| LLaMA-2-13B (4096 tokens) | ~3.125 GB | 更大的模型更严重 |
| DeepSeek-V3 (671B, FP16) | 1,342 GB | 超大模型的极端情况 |

**核心矛盾**：GPU内存几十GB，模型参数占65%，剩余空间只能容纳几十个并发请求的KV cache。内存管理不善则进一步缩减。

---

## 第二部分：现有系统的三大内存浪费

### 浪费来源

现有系统（FasterTransformer、Orca等）要求KV cache存储在**连续内存**中：

**1. 预留浪费（Reservation）**
- 按最大可能长度预分配（如2048 tokens）
- 实际生成可能只有几十个token
- 预留期间其他请求无法使用这些内存

**2. 内部碎片（Internal Fragmentation）**
- 请求结束后才知道实际长度
- 预分配与实际使用的差距 = 纯浪费
- 例：预分配2048 slots，实际用10个，2038个slots浪费

**3. 外部碎片（External Fragmentation）**
- 不同请求预分配大小不同
- 释放后留下大小不一的内存间隙
- 新请求可能无法放入这些碎片化空间

### 实测数据

| 系统 | 有效内存利用率 | 浪费比例 |
|-----|-------------|---------|
| Orca (Max预分配) | 20.4% | 79.6% |
| Orca (2的幂预分配) | 26.8% | 73.2% |
| Orca (Oracle, 已知长度) | 38.2% | 61.8% |
| **vLLM** | **96.3%** | **3.7%** |

> **关键发现：现有系统 60-80% 的KV Cache内存被浪费了！**

---

## 第三部分：PagedAttention 核心算法

### 灵感来源：操作系统虚拟内存

操作系统在几十年前就解决了类似问题 — 进程需要连续的逻辑地址空间，但物理内存可能是碎片化的。解决方案：**虚拟内存 + 分页**。

### 概念映射

| 操作系统 | PagedAttention | 作用 |
|---------|---------------|------|
| 页（Page） | KV Block | 固定大小的内存单元 |
| 字节（Byte） | Token | 最小数据单位 |
| 进程（Process） | Request（请求） | 内存使用者 |
| 页表（Page Table） | Block Table（块表） | 逻辑→物理映射 |
| 虚拟地址 | 逻辑KV块编号 | 连续的逻辑视图 |
| 物理页帧 | 物理KV块 | 实际GPU内存位置 |
| 空闲页链表 | Free Block Pool | 管理可用内存块 |
| 写时复制 (CoW) | Copy-on-Write | 共享+延迟复制 |
| 交换空间 (Swap) | CPU RAM Swap | 内存不足时卸载 |

### 算法工作流程

**Step 1: 初始分配**
```
请求到达 → 仅分配prompt所需的KV块数量
（不预留最大长度！）

例：prompt = 7 tokens, block_size = 4
→ 分配 2 个逻辑块（block 0: 4 tokens, block 1: 3 tokens + 1空位）
→ 映射到 2 个物理块（可以不连续）
```

**Step 2: 逐步扩展**
```
每生成一个新token：
  if 当前最后一个块还有空位:
    → 直接写入，更新块表的filled计数
  else:
    → 分配新的物理块
    → 建立新的逻辑→物理映射
    → 写入新token的KV cache
```

**Step 3: 注意力计算**
```
PagedAttention 内核执行时：
  1. 查询当前请求的块表
  2. 按逻辑顺序遍历所有KV块
  3. 从各个（不连续的）物理位置加载K和V
  4. 维护跨块的running softmax归一化
  5. 产生与标准连续注意力完全相同的结果
```

**Step 4: 释放回收**
```
请求完成 → 释放所有物理块 → 归还到空闲块池
→ 可被新请求立即复用
```

### 为什么只有不到4%的浪费？

- **无外部碎片**：所有块大小相同，任何空闲块都能满足任何请求
- **无预留浪费**：按需分配，用多少分多少
- **内部碎片极小**：仅在最后一个块中（最多 block_size - 1 个slot浪费）

---

## 第四部分：高级特性

### 4.1 Copy-on-Write（写时复制）— 并行采样

**场景**：同一个prompt生成多个不同输出（如代码补全生成多个候选）

**工作原理**：
```
Sample A1 和 A2 共享同一个prompt
→ prompt 的逻辑块都映射到相同的物理块
→ 物理块引用计数 = 2

当 A1 需要写入共享块时：
→ 检测到引用计数 > 1
→ 复制该物理块到新位置
→ A1 的逻辑块映射到新物理块
→ 原物理块引用计数减1
→ A2 继续使用原块
```

**节省效果**：prompt部分的KV cache完全共享，仅在分歧点复制一个块。

### 4.2 Beam Search 的动态共享

**场景**：翻译等任务中寻找top-k最优输出

**传统系统的问题**：每个beam候选维护独立的KV cache副本，大量重复复制。

**vLLM方案**：
- 不同beam候选在block粒度上**动态共享**物理块
- 共享模式随解码推进而变化（类似OS的fork进程树）
- 仅在写入共享块时触发copy-on-write
- 节省高达 **55%** 的内存

### 4.3 共享前缀（Shared Prefix）

**场景**：多个请求共享相同的系统提示（system prompt）

```
系统提示："你是一个翻译助手..."（共享前缀）
用户A："翻译：cheese"
用户B："翻译：I love you"

→ 系统提示的KV cache物理块被预缓存
→ 所有请求直接映射到缓存块（标记为CoW）
→ 仅需计算用户特定输入部分
```

**类比**：OS中多进程共享动态链接库（.so/.dll）

### 4.4 连续批处理（Continuous Batching）

传统批处理：等整个batch完成才处理下一批 → 快的请求等慢的

vLLM连续批处理：
```
每次迭代：
  → 已完成的请求立即移出batch
  → 新请求立即加入batch
  → GPU利用率保持 90%+
```

**效果**：2-4倍吞吐量提升，延迟可预测

### 4.5 抢占与交换

当GPU内存不足时：

| 策略 | 机制 | 何时使用 |
|-----|------|---------|
| Swapping | KV块从GPU复制到CPU内存 | CPU-GPU带宽充足 |
| Recomputation | 释放KV cache，重新计算 | GPU计算能力强 |

调度策略：
- **FCFS**（先来先服务）
- **全有或全无驱逐**（一个序列的所有块一起驱逐）
- **组调度**（同一请求的多个序列一起处理）

### 4.6 分布式执行

- 支持 Megatron-LM 张量模型并行
- SPMD执行模式
- 集中式调度器 + 分布式GPU Worker
- KV cache管理跨GPU完全同步

---

## 第五部分：性能评估

### 吞吐量对比

| 场景 | vLLM vs HuggingFace | vLLM vs FasterTransformer | vLLM vs Orca |
|------|--------------------|--------------------------|--------------|
| 通用 | **24x** | **14-24x** | **2-4x** |
| 长序列 | 更高 | 更高 | 更高 |
| 复杂解码 | 更高 | 更高 | 更高 |

### 真实部署数据 (LMSYS Chatbot Arena)

- 日均处理 **30,000** 请求，峰值 **60,000** 请求
- 相比HF初始后端：吞吐量提升 **30x**
- GPU需求降低 **50%**

### 内存效率

- 内存浪费从 60-80% 降至 **不到4%**
- 内存节省 **24-48%**
- H100 GPU上 token生成速度 **500+ tokens/s**

---

## 第六部分：生态与影响

### 支持的模型和功能

- 支持 **100+** 模型（LLaMA, GPT, OPT, Mistral, Qwen 等）
- 量化支持：AWQ, GPTQ, FP8（内存再降2-4x）
- 推测解码：小模型提议 + 大模型验证，速度翻倍
- 视觉语言模型支持（LLaVA等）
- OpenAI 兼容API
- LoRA 热加载

### 代码示例

```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# 批量推理
prompts = ["Hello, world!", "Why is the sky blue?"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

启动API服务器：
```bash
vllm serve meta-llama/Llama-2-7b-hf --port 8000
```

### 后续影响

PagedAttention 开创的范式影响了整个LLM推理领域：

| 后续工作 | 在PagedAttention基础上的发展 |
|---------|--------------------------|
| Mooncake | KV Cache分离式架构 + 多层存储 |
| DistServe | Prefill/Decode分离到不同GPU |
| SGLang | RadixAttention 前缀树缓存 |
| FlashInfer | 自定义注意力内核引擎 |
| StreamingLLM | 注意力汇聚 + 滑动窗口 |

---

## 第七部分：关键要点速记

### 一句话总结
> PagedAttention 将OS虚拟内存的分页思想引入LLM推理，通过非连续KV Cache存储、写时复制和按需分配，将内存利用率从20%提升到96%，吞吐量提升2-24倍。

### 五个核心创新
1. **分页存储** — KV cache分块，非连续存储，消除碎片
2. **按需分配** — 不预留最大长度，用多少分多少
3. **写时复制** — 并行采样/beam search的高效内存共享
4. **连续批处理** — 迭代级调度，GPU利用率90%+
5. **多级内存管理** — GPU↔CPU交换 + 重计算恢复

### 为什么这篇工作如此重要？
- **实际部署已验证**：LMSYS Chatbot Arena日均60K请求
- **成为事实标准**：几乎所有现代LLM serving系统都基于PagedAttention
- **开源生态**：vLLM项目在GitHub上被广泛采用
- **跨学科创新**：证明OS思想可以解决AI系统问题
- **经济价值**：2-4倍吞吐量 = 50-75%成本降低

---

## 资料来源

1. [vLLM 官方博客](https://vllm.ai/blog/vllm) — 项目介绍与性能数据
2. [Paged Attention from First Principles](https://hamzaelshafie.bearblog.dev/paged-attention-from-first-principles-a-view-inside-vllm/) — 原理深度解析
3. [DEV Community: vLLM Explained](https://dev.to/jaskirat_singh/vllm-explained-how-pagedattention-makes-llms-faster-and-cheaper-785) — 功能全面总结
4. [Woosuk Kwon MLSys Seminar Talk](https://www.youtube.com/watch?v=Oq2SN7uutbQ) — 作者本人演讲
5. [Woosuk Kwon CMU 2025 课程讲义](https://llmsystem.github.io/llmsystem2025spring/) — 最新教学材料
6. [SOSP 2023 原论文](https://arxiv.org/abs/2309.06180) — 完整技术细节

---

*整理日期：2026-03-21*
