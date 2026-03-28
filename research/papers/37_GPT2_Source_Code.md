# GPT-2 Source Code Repository

**项目:** GPT-2 (Language Models are Unsupervised Multitask Learners)
**作者:** Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever (OpenAI)
**发表:** 2019
**论文:** https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
**GitHub:** https://github.com/openai/gpt-2

---

## 一、仓库概述

OpenAI 官方发布的 GPT-2 代码仓库，包含模型定义、文本生成和条件生成的完整实现。

### 仓库结构
```
gpt-2/
├── src/
│   ├── model.py          # GPT-2 模型定义 (TensorFlow)
│   ├── sample.py          # 采样/生成逻辑
│   ├── encoder.py         # BPE tokenizer 实现
│   ├── generate_unconditional_samples.py  # 无条件生成
│   └── interactive_conditional_samples.py # 交互式条件生成
├── models/                # 预训练模型权重
│   ├── 124M/             # 小模型 (117M参数)
│   ├── 355M/             # 中模型
│   ├── 774M/             # 大模型
│   └── 1558M/            # 超大模型 (1.5B参数)
└── requirements.txt
```

## 二、GPT-2 模型关键特点

| 特性 | 说明 |
|------|------|
| 架构 | Decoder-only Transformer |
| 参数量 | 117M / 345M / 762M / 1542M |
| 训练数据 | WebText (40GB 网页文本, ~8M 文档) |
| 上下文长度 | 1024 tokens |
| 词汇表 | 50,257 (BPE) |
| 层数 (1.5B) | 48 层, 1600 维, 25 个头 |

## 三、架构细节 (vs 原始 Transformer)

GPT-2 相比 "Attention Is All You Need" 的改进：

1. **仅 Decoder**: 去掉 Encoder，纯自回归生成
2. **Pre-Layer Normalization**: LayerNorm 移到子层输入前（而非输出后）
3. **更大上下文**: 512 → 1024 tokens
4. **BPE Tokenizer**: Byte-level BPE，可处理任意文本
5. **权重初始化**: 残差层权重按 `1/√N` 缩放（N = 残差层数）

```python
# 核心模型结构 (简化)
def model(X, past=None):
    # Token embedding + Position embedding
    h = wte[X] + wpe[positions]

    # N 个 Transformer block
    for block in transformer_blocks:
        h = block(h)  # LayerNorm → Attention → LayerNorm → FFN

    # 语言模型头
    logits = h @ wte.T  # 权重共享
    return logits
```

## 四、学习价值

### 代码阅读重点
1. **`src/model.py`** — 理解 Transformer decoder 的完整实现
2. **`src/sample.py`** — 学习 top-k 采样和温度控制
3. **`src/encoder.py`** — 理解 BPE tokenizer 的工作原理

### 推荐学习路径
1. 先阅读 "Attention Is All You Need" 理解基础架构
2. 对照 GPT-2 代码理解 Decoder-only 的实际实现
3. 运行生成示例，调整参数观察效果
4. 阅读 Sparse Transformer 论文理解长序列处理优化

## 五、相关社区实现

- **Hugging Face Transformers**: `transformers.GPT2LMHeadModel` (PyTorch)
- **nanoGPT** (Andrej Karpathy): https://github.com/karpathy/nanoGPT — 最简洁的 GPT 训练代码
- **minGPT** (Andrej Karpathy): https://github.com/karpathy/minGPT — 教学用最小实现

## 六、历史意义

GPT-2 是 LLM 发展史上的重要里程碑：
- 首次展示**零样本学习 (zero-shot)** 在多任务上的强大能力
- 证明了 **"规模即能力"** — 更大模型 = 更强泛化
- 引发了关于 AI 安全和负责任发布的广泛讨论
- 直接导向了 GPT-3 → ChatGPT → GPT-4 的发展路径
