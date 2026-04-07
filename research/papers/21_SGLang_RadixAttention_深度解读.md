# SGLang / RadixAttention 论文深度解读

**论文:** SGLang: Efficient Execution of Structured Language Model Programs
**作者:** Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, Ying Sheng
**机构:** Stanford University, UC Berkeley, Shanghai Jiao Tong University, Texas A&M, NVIDIA
**arXiv:** 2312.07104 (2024年6月更新)
**代码:** https://github.com/sgl-project/sglang

---

## 一、解决什么问题？

现代LLM应用越来越复杂：多步骤Agent、Tree-of-Thought、RAG流水线、few-shot learning都需要**多次LLM调用**，形成"LM程序"：

```
LM程序的两大特征:
  1. 包含多个LLM调用，中间穿插控制流
  2. 接收结构化输入，产生结构化输出

当前系统的两大痛点:

痛点1: 编程繁琐
  → 需要手动处理字符串拼接、并行控制、输出解析
  → 即使简单的few-shot程序也需要大量模板代码
  → 调试困难，可读性差

痛点2: 执行低效
  → 同一批次的请求经常共享相同前缀(系统提示、few-shot例子)
  → 但这些共享前缀的KV cache每次请求都被重新计算
  → vLLM等系统在每个请求结束后直接丢弃KV cache
  → 无法自动处理复杂的树形共享模式
```

**SGLang的目标：提供统一的编程语言 + 运行时框架，自动识别并复用共享KV cache，显著提升LM程序的执行效率。**

---

## 二、核心方法/关键发现

### SGLang前端语言

SGLang是嵌入Python的领域专用语言，提供6类原语：

```python
# SGLang核心原语:
s += system("You are a helpful assistant.")  # extend: 添加文本
s += user("Hello!")
result = s.assistant(gen("reply"))           # gen: LLM生成
choice = s.assistant(select("yes|no"))       # select: 选择最高概率项

# 并行控制:
forks = s.fork(3)                            # fork: 创建并行副本
for f in forks:
    f += user("Evaluate dimension X")
    f += assistant(gen("judgment"))
s.join(forks)                                # join: 合并

# 多模态:
s += user(image("path.jpg") + "Describe this") # image/video: 多模态输入
```

### RadixAttention：KV Cache的自动复用

核心创新：**用基数树(Radix Tree)管理KV cache，自动处理所有前缀共享模式**

```
关键洞察:
  1. KV cache只依赖于前缀token → 共享前缀 = 可复用KV cache
  2. LM程序中的共享模式非常丰富:
     - Few-shot例子共享
     - 系统提示共享
     - Multi-turn chat历史共享
     - Tree-of-Thought搜索历史共享
     - Self-consistency采样

传统做法:
  每个请求完成 → KV cache立即丢弃
  → 下一个请求如果有相同前缀 → 重新计算

RadixAttention做法:
  维护一个全局Radix Tree，存储所有请求的KV cache
  每个节点 = 一段token序列 + 对应的KV cache tensor
  使用LRU淘汰策略管理内存
```

### Radix Tree运作示例

```
时间点1: 请求"You are a helpful assistant. User: Hello! Assistant: Hi!"
  树: [root] → [You are helpful...Hello!Hi!]  (单节点)

时间点2: 新请求共享前缀"You are a helpful assistant."，新增"User: How are you?"
  树: [root] → [You are helpful assistant.] → [Hello!Hi!]
                                              → [How are you? ...]
  系统提示前缀被共享 → KV cache复用!

时间点3: Few-shot查询，共享examples
  树: [root] → [system prompt] → [Q1:A1, Q2:A2, Q3:A3] → [new question]
  → Few-shot examples只计算一次!

时间点4: Self-consistency采样，从同一问题分叉
  树: [root] → [system+Q3] → [Answer1]
                            → [Answer2]
                            → [Answer3]
  → 问题前缀只计算一次，3个答案并行生成

LRU淘汰:
  当内存不足时，优先淘汰最近最少使用的叶节点
  → 淘汰叶子后，其父节点(公共前缀)可继续服务其他请求
```

### 缓存感知调度 (Cache-Aware Scheduling)

```
问题: 等待队列中有多个请求时，执行顺序影响缓存命中率

朴素策略 (FCFS): 先到先服务
  → 频繁切换不同前缀的请求 → cache thrashing → 命中率低

SGLang策略: 最长共享前缀优先
  → 选择与当前RadixTree共享最长前缀的等待请求
  → 最大化已缓存KV的利用率

理论保证 (Theorem 3.1):
  对于一批请求，按radix树深度优先搜索(DFS)顺序执行
  可以实现最优缓存命中率
  (当cache大小 ≥ 最长请求长度时，DFS = 最优)

实测: SGLang的缓存感知调度达到Oracle最优命中率的96%
```

---

## 三、技术细节

### Radix Tree数据结构

```
关键设计:
  - 边标签: 可以是任意长度的token序列 (不限单字符)
  - 节点内容: 对应edge上tokens的KV cache tensor
  - 内存格式: non-contiguous paged layout (与vLLM兼容)
               每页等于1个token的KV cache

操作复杂度:
  前缀匹配: O(前缀长度)
  插入: O(插入长度)
  淘汰: O(1) per eviction (LRU链表)

Reference Counting:
  每个节点维护引用计数 = 当前使用该节点的活跃请求数
  节点引用计数=0 → 可以被LRU淘汰
  确保正在使用的节点不被淘汰
```

### 多模态支持

```
图像输入:
  → 对图像内容计算哈希值作为RadixTree的key
  → 相同图像的多个问题共享image token的KV cache

视频输入:
  → 类似图像，按帧计算哈希
  → 视频理解任务中多问题共享帧KV cache大幅减少计算
```

### 压缩有限状态机 (Compressed FSM)

```
第二个核心优化: 结构化输出的快速解码

问题:
  {"summary": "...", "grade": "A"}
  这类固定格式的常量字符串需要多步解码
  即使下一个token完全确定，系统仍逐token解码

SGLang方案:
  将正则表达式转换为FSM
  压缩FSM: 合并所有单一转移边(singular transitions)
  → 一段确定性字符串变成一步解码

效果:
  {"summary": ": "} 这段字符串: 原来13步 → 压缩后1步
  JSON解码吞吐量提升: 1.6×
```

### 前端运行时协同

```
SGLang前端-运行时协作:

前端(解释器):
  → 使用fork时，先把共享前缀作为"Frontend Hint"发送给运行时
  → 运行时提前把该前缀插入RadixTree
  → 后续fork的多个分支可以立即命中缓存

运行时:
  → 接收前端hint → 预处理前缀
  → 缓存感知调度确保高命中率
  → 对外提供OpenAI兼容API

这种前端-运行时协同设计是其他系统没有的
```

---

## 四、实验结果

### 端到端吞吐量 (Llama-7B, vs vLLM/Guidance/LMQL)

```
SGLang在多种workload上的归一化吞吐量 (以最慢为1×基准):

Workload        SGLang vs vLLM   主要原因
────────────────────────────────────────────
MMLU (5-shot)    6.4×  ↑          RadixAttention复用few-shot KV
ReAct Agent      4.1×  ↑          Agent模板KV复用
Tree-of-Thought  3.2×  ↑          搜索历史KV复用
JSON Decoding    2.1×  ↑          压缩FSM快速解码
Multi-Turn(short) 1.3× ↑          对话历史复用
Multi-Turn(long)  ~1×              长输出时decode主导，共享帮助有限
DSPy RAG         5.0×  ↑          文档上下文KV复用

总结: 最高吞吐量提升 6.4×, 延迟降低最高 3.7×
```

### 缓存命中率分析

```
不同workload的实际缓存命中率:

MMLU:         99% (5-shot examples几乎永远缓存)
Few-shot RAG: 98% (文档KV长期有效)
ReAct Agent:  90% (Agent模板高度复用)
Multi-turn:   50%~80% (依赖会话长度)

理论最优 vs 实际:
  SGLang缓存感知调度达到Oracle最优的 96%
```

### 多模态模型测试

```
LLaVA视觉模型:

                        原始实现    SGLang
LLaVA-v1.5-7B(image)  0.18 img/s  1.15 img/s  → 6.4×
LLaVA-NeXT-34B(video)  0.02 fps    0.10 fps    → 5×

原因: 相同图像的多问题共享image token KV
     在llava-bench-in-the-wild中图像被多次查询
```

### 生产部署数据

```
Chatbot Arena实际部署数据 (1个月):

LLaVA-Next-34B:  52.4% RadixAttention缓存命中率
Vicuna-33B:      74.1% 缓存命中率

来源: 常见系统提示、频繁重用的示例图像、多轮对话历史
效果: Vicuna-33B首token延迟平均降低 1.7×
```

### RadixAttention开销测试

```
在ShareGPT (无缓存机会的数据集)上:
  运行100个请求: 74.3秒
  RadixTree管理开销: 仅0.2秒 = 0.3%
  
→ 当没有缓存复用机会时，开销几乎为零
→ 可以默认开启，无需手动配置
```

---

## 五、核心启示与局限

### 核心启示

```
1. KV cache是可复用的系统资源，应像缓存一样管理:
   传统视角: KV cache = 单请求的临时数据
   SGLang视角: KV cache = 系统级共享缓存
   → 从"垃圾回收"思维转变为"缓存管理"思维

2. 前缀共享模式在实际应用中极为普遍:
   系统提示: 所有请求共享
   Few-shot例子: 同类请求共享
   Multi-turn历史: 同会话共享
   Agent模板: 同类Agent共享
   → 利用好这个属性可以获得数倍的吞吐量提升

3. 前端-运行时协同设计放大了优化效果:
   前端提供语义信息(fork的共享部分)
   运行时可以提前预处理，减少调度延迟
   纯运行时系统缺乏这些语义信息

4. 缓存感知调度是关键:
   激进缓存 + 随机调度 = 缓存命中率很低
   激进缓存 + 感知调度 = 接近最优命中率
   两者缺一不可

5. 多token解码对结构化输出很重要:
   JSON、代码等格式有大量可预测的固定段
   压缩FSM将其折叠为单步解码 → 1.6×加速
   与KV复用协同效应良好
```

### 局限性

```
1. 缓存效益依赖工作负载特性:
   随机无关的请求 → 几乎无共享 → 收益小
   需要应用层面有共享前缀才能受益
   纯chatbot场景收益有限

2. LRU淘汰可能导致饥饿:
   缓存感知调度优先处理有cache的请求
   没有共享前缀的请求可能被长期饿死
   论文承认这是未来工作

3. 对API模型(GPT-4等)支持有限:
   API speculative execution只能优化部分场景
   GPT-4的tokenizer和缓存机制不透明

4. 多GPU分布式场景的复杂性:
   Tensor并行时，每个GPU维护一个sharded KV cache
   Data并行时，需要worker间协调
   复杂拓扑结构的RadixTree同步尚未完善

5. RadixAttention与其他KV压缩技术的结合:
   论文提到可以结合量化、稀疏attention
   但工程实现上需要更多工作
```

---

## 六、在知识体系中的位置

```
KV Cache管理的不同维度:

  维度1: 单请求内部 (减少cache大小)
    H2O, Quest, InfiniGen → 在一个请求内选择性保留/加载KV

  维度2: 跨请求共享 (RadixAttention的贡献)
    → 不同请求如果有共享前缀，复用同一份KV cache
    → 从"per-request"到"system-wide"的视角转变

  维度3: 跨存储层次
    PagedAttention: GPU内无碎片管理
    FlexGen/InfiniGen: GPU-CPU offloading
    KVSwap: CPU-Disk offloading

SGLang的独特之处:
  → 将KV cache复用从"单场景优化"提升为"系统级基础设施"
  → 通过前端语言与运行时协同提供编程友好性
  → 已在生产环境(Chatbot Arena)验证有效性
  → 已被vLLM等主流框架借鉴采用
```

## 一句话总结

> **SGLang提出RadixAttention，将KV cache管理从单请求临时存储升级为系统级LRU缓存，通过基数树数据结构自动识别并复用跨请求的共享前缀KV，配合缓存感知调度算法和压缩有限状态机，在few-shot学习、Agent控制、Tree-of-Thought等复杂LM程序中实现最高6.4×的吞吐量提升，并在Chatbot Arena生产部署中验证了52-74%的实际缓存命中率。**

---

*解读日期：2026-04-07*
