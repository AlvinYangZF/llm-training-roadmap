# StreamingLLM 论文深度解读

**论文:** Efficient Streaming Language Models with Attention Sinks
**作者:** Guangxuan Xiao 等 (MIT, Meta AI, CMU, NVIDIA)
**会议:** ICLR 2024
**arXiv:** 2309.17453

---

## 一、解决什么问题？

LLM在流式/长对话场景（如全天候聊天机器人）中有两个致命问题：

1. **KV cache无限增长** — 对话越长，缓存越大，内存终将耗尽
2. **长度外推失败** — 模型在超过训练长度后性能崩溃

现有方案都不理想：

```
方案A: Dense Attention (保留全部KV)
  内存: O(T²) 无限增长 → 最终OOM
  性能: 超过训练长度后崩溃
  PPL:  5641✗

方案B: Window Attention (只保留最近L个token)
  内存: O(L) 恒定 ✓
  性能: 一旦丢掉初始token就崩溃！
  PPL:  5158✗

方案C: Sliding Window + Recomputation (每次重算)
  内存: O(L) 恒定 ✓
  性能: 很好 ✓
  速度: O(TL²) 极慢，不实用 ✗
  PPL:  5.43✓

方案D: StreamingLLM (本文)
  内存: O(L) 恒定 ✓
  性能: 稳定 ✓
  速度: O(TL) 快 ✓
  PPL:  5.40✓
```

## 二、核心发现：Attention Sink（注意力汇聚）

论文最重要的发现是一个令人惊讶的现象：

> **LLM在所有层、所有头中，都会给最初几个token分配异常高的注意力分数 — 即使这些token在语义上毫不重要。**

```
Attention分布可视化 (Llama-2-7B):

Layer 0-1 (浅层):  局部注意力为主，最近token得到更多关注
Layer 2+ (深层):  第一个token获得压倒性的注意力！

  Token位置:  0   1   2   3   4   5   6   7 ...
  注意力:   ████ █  ·  ·  ·  ·  · ███
            ↑                        ↑
         Attention Sink          最近的token
         (注意力汇聚)
```

## 三、为什么会有Attention Sink？

原因出在Softmax的数学特性：

```
Softmax(xᵢ) = exp(xᵢ) / Σexp(xⱼ)

关键：softmax要求所有注意力分数之和 = 1
```

当模型对某个位置"不需要关注任何token"时，它仍然必须把注意力分配出去（总和=1）。模型学会了把这些"多余的注意力"倾倒到初始token上。

**为什么是初始token而不是其他token？**

```
自回归LM中，token的可见性：
  Token 0: 被所有后续token看见 → 最容易成为"垃圾桶"
  Token 1: 被除了token 0之外的所有token看见
  Token N: 只被后面少数token看见

→ 初始token是全局可见的，自然成为attention的"垃圾桶"
```

## 四、验证实验：语义无关，位置才重要

```
实验：把初始4个token替换成换行符"\n"

  原始 (4个真实token + 1020个最近token): PPL = 5.40
  替换 (4个"\n" + 1020个最近token):     PPL = 5.60  ← 几乎一样好！

→ 证明模型不关心初始token的内容，只需要它们的位置存在
→ 这是纯粹的"注意力垃圾桶"效应
```

## 五、StreamingLLM算法

极其简洁的设计 — KV cache分两部分：

```
┌─────────────┬─ · · · ─┬───────────────────┐
│ Attention   │ 被淘汰的 │   Rolling KV      │
│ Sink (4个)  │  tokens  │   Cache (最近L个)  │
└─────────────┴─ · · · ─┴───────────────────┘
     ↑                          ↑
  保留初始4个token         保留最近L个token的KV
  稳定softmax分布         保持局部上下文信息

总cache大小 = 4 + L = 恒定！
```

### 生成过程

```
生成Token 7:  cache = [0, 1, 2, 3, | 4, 5, 6, 7]
                       sink tokens    recent tokens

生成Token 8:  cache = [0, 1, 2, 3, | 5, 6, 7, 8]   ← token 4被淘汰
                       sink tokens    recent tokens

生成Token 9:  cache = [0, 1, 2, 3, | 6, 7, 8, 9]   ← token 5被淘汰
                       sink tokens    recent tokens

→ 永远保持恒定大小的cache
→ 永远不丢失初始sink tokens
```

### 位置编码的关键细节

```
cache中的token位置: [0, 1, 2, 3, 6, 7, 8, 9]  ← 原始位置
StreamingLLM重映射: [0, 1, 2, 3, 4, 5, 6, 7]  ← cache内连续位置

→ 位置编码按cache内的位置，不是原文位置
→ 避免位置编码出现不连续的"跳跃"
```

## 六、更进一步：训练时加入Sink Token

论文还提出了一个前瞻性建议：

```
现有模型: 需要保留4个初始token作为sink（因为训练时没有专门的sink）
改进方案: 预训练时在每个样本开头加一个专用的可学习sink token

效果对比 (160M参数模型):
  Vanilla (无sink):           PPL = 27.87 (0+1024 cache时)
  Zero Sink (softmax变体):    PPL = 19.90
  Learnable Sink (本文建议):  PPL = 18.01  ← 最好

→ 只需1个专用sink token就够了（而不是4个初始token）
→ 不损害模型在标准NLP基准上的表现
```

## 七、性能结果

### 超长文本稳定性（4百万token！）

在PG19测试集（100本书连接）上，StreamingLLM在所有模型家族和规模上保持稳定：

| 模型 | 400万token后perplexity稳定？ |
|------|--------------------------|
| Llama-2 (7B, 13B, 70B) | ✓ 稳定 |
| Falcon (7B, 40B) | ✓ 稳定 |
| Pythia (2.8B - 12B) | ✓ 稳定 |
| MPT (7B, 30B) | ✓ 稳定 |

### 速度对比

```
vs 唯一可行的替代方案（Sliding Window + Recomputation）:

StreamingLLM 快 22.2×！

原因：Recomputation每生成一个token都要重算整个窗口内的attention
      StreamingLLM只需增量更新cache
```

### 流式问答（模拟真实聊天）

| 模型 | Dense | Window | StreamingLLM | One-shot基线 |
|------|-------|--------|-------------|------------|
| Llama-2-7B-Chat | OOM | 3.58% | **71.34%** | 71.25% |
| Llama-2-13B-Chat | OOM | 0.25% | **80.89%** | 78.16% |
| Llama-2-70B-Chat | OOM | 0.12% | **91.37%** | 91.29% |

Dense直接OOM，Window几乎随机，StreamingLLM接近逐条处理的基线。

### Attention Sink数量分析

| Cache配置 | Falcon-7B | MPT-7B | Pythia-12B | Llama-2-7B |
|----------|-----------|--------|------------|------------|
| 0+cache (无sink) | 17.90 | 460.29 | 21.62 | 3359.95 |
| 1+cache | 12.12 | 14.99 | 11.95 | 11.88 |
| 2+cache | 12.12 | 15.00 | 12.09 | 10.51 |
| 4+cache | **12.12** | **14.99** | **12.09** | **9.59** |
| 8+cache | 12.12 | 14.98 | 12.02 | 9.54 |

→ 4个sink tokens几乎达到最优，更多sink带来的收益极小

## 八、局限性

```
StreamingLLM 能做的：
  ✓ 无限长度的流式生成
  ✓ 恒定内存使用
  ✓ 保持perplexity稳定
  ✓ 不需要微调

StreamingLLM 不能做的：
  ✗ 不扩展真正的上下文窗口（看不到被淘汰的中间token）
  ✗ 不能回忆很久以前的对话内容
  ✗ 适合"最近信息最重要"的场景，不适合需要全局记忆的场景
```

## 九、在知识体系中的位置

```
论文1: PagedAttention — 如何高效管理KV cache的内存分配
论文2: FlashAttention — 如何高效计算单次attention
论文3: StreamingLLM  — KV cache可以丢弃大部分内容！

三者组合：
  FlashAttention (高效计算)
    + PagedAttention (高效分配)
      + StreamingLLM (高效淘汰)
        = 内存恒定、计算高效、无限长度的LLM推理
```

## 十、一句话总结

> **StreamingLLM发现LLM会将"多余的注意力"倾倒到初始token上（Attention Sink现象），据此提出只保留4个初始token + 最近token的滚动cache策略，用恒定O(L)内存实现无限长度的稳定推理，比重计算方案快22倍。**

---

*解读日期：2026-03-25*
