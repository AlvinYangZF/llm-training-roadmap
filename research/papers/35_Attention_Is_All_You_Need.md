# Attention Is All You Need

**论文:** Attention Is All You Need
**作者:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google Brain, Google Research, University of Toronto)
**发表:** NeurIPS 2017
**arXiv:** 1706.03762
**链接:** https://arxiv.org/abs/1706.03762

---

## 一、核心贡献

提出了 **Transformer** 架构 — 完全基于注意力机制，彻底抛弃了 RNN 和 CNN。这篇论文是现代大语言模型（GPT、BERT、LLaMA 等）的奠基之作。

## 二、解决什么问题？

传统序列建模（RNN/LSTM）的两大瓶颈：
- **顺序计算**: 必须逐步处理序列，无法并行化，训练慢
- **长距离依赖**: 随着序列变长，梯度消失/爆炸，难以捕捉远距离关系

## 三、核心架构

### Encoder-Decoder 结构

```
输入序列 → [Encoder × 6] → 编码表示
                                 ↓
输出序列 → [Decoder × 6] → 预测输出
```

### 关键组件

1. **Multi-Head Self-Attention (多头自注意力)**
   - 将 Q、K、V 分成 h 个头，每个头独立计算注意力
   - 允许模型同时关注不同位置的不同表示子空间
   - `Attention(Q, K, V) = softmax(QK^T / √d_k) V`

2. **Position-wise Feed-Forward Networks**
   - 两层线性变换 + ReLU: `FFN(x) = max(0, xW₁ + b₁)W₂ + b₂`

3. **Positional Encoding (位置编码)**
   - 使用正弦/余弦函数编码位置信息
   - `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
   - `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`

4. **Residual Connection + Layer Normalization**
   - 每个子层都有残差连接和层归一化

## 四、Scaled Dot-Product Attention

```
        Q · K^T
Attention = softmax(─────────) · V
              √d_k
```

- 缩放因子 `√d_k` 防止点积过大导致 softmax 梯度消失
- 相比加性注意力，点积注意力更快且更节省空间

## 五、为什么重要？

| 特性 | RNN/LSTM | Transformer |
|------|----------|-------------|
| 并行化 | 不可以 | 完全并行 |
| 最大路径长度 | O(n) | O(1) |
| 每层复杂度 | O(n·d²) | O(n²·d) |
| 训练速度 | 慢 | 快很多 |

## 六、实验结果

- **WMT 2014 英德翻译**: 28.4 BLEU（当时 SOTA）
- **WMT 2014 英法翻译**: 41.0 BLEU（单模型 SOTA）
- 训练成本仅为之前最佳模型的 **1/4**

## 七、深远影响

这篇论文催生了整个现代 NLP/LLM 生态：
- **BERT** (2018): 仅用 Encoder，双向预训练
- **GPT 系列** (2018-2024): 仅用 Decoder，自回归生成
- **Vision Transformer (ViT)**: 将 Transformer 扩展到视觉领域
- **所有现代 LLM** (LLaMA, Claude, Gemini 等) 都基于 Transformer 架构
