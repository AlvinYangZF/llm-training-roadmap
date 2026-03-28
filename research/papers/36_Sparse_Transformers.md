# Generating Long Sequences with Sparse Transformers

**论文:** Generating Long Sequences with Sparse Transformers
**作者:** Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever (OpenAI)
**发表:** 2019
**arXiv:** 1904.10509
**链接:** https://arxiv.org/abs/1904.10509

---

## 一、核心贡献

提出 **Sparse Transformer** — 通过稀疏注意力模式将标准 Transformer 的 O(n²) 注意力复杂度降低到 O(n√n)，使其能够处理数万 token 的长序列。

## 二、解决什么问题？

标准 Transformer（Attention Is All You Need）的自注意力是**全连接**的：
- **内存**: O(n²) — 序列长度 n=16384 时，注意力矩阵需要 ~1GB
- **计算**: O(n²) — 序列越长，计算量平方增长
- **实际限制**: 难以处理超过几千 token 的序列

## 三、核心思想 — 稀疏注意力模式

### 标准 (Dense) Attention
每个 token 关注所有其他 token：
```
● ● ● ● ● ● ● ●
● ● ● ● ● ● ● ●
● ● ● ● ● ● ● ●  → O(n²) 连接
● ● ● ● ● ● ● ●
```

### Sparse Attention — Strided Pattern
将注意力分解为两种互补模式：

1. **Local Attention (局部注意力)**
   - 每个 token 只关注前面 √n 个相邻 token
   - 捕捉局部上下文

2. **Strided Attention (跨步注意力)**
   - 每隔 √n 个位置关注一个 token
   - 捕捉全局长距离依赖

```
头1 (局部):     ○ ○ ○ ● ● ● ● ●  ← 只看附近
头2 (跨步):     ● ○ ○ ● ○ ○ ● ○  ← 间隔采样
组合效果:       任意两个token最多经过2步可达
```

### Fixed Pattern（适用于图像/结构化数据）
- 特定行/列的 token 作为"汇总节点"
- 其他 token 通过汇总节点间接通信

## 四、关键技术细节

### 分解注意力 (Factorized Attention)
将完整的注意力矩阵分解为 p 个稀疏头的乘积：
- 每个头只有 O(n/p) 个非零连接
- p 个头组合起来覆盖所有位置对
- 总复杂度: O(n · n^(1/p)) ，当 p=2 时为 O(n√n)

### 其他优化
- **梯度检查点 (Gradient Checkpointing)**: 重计算中间激活值，降低内存
- **混合精度训练**: FP16 计算 + FP32 累加
- **高效 CUDA 内核**: 自定义稀疏注意力 GPU 实现

## 五、实验结果

在多种模态上验证：

| 任务 | 序列长度 | 结果 |
|------|---------|------|
| CIFAR-10 图像生成 | 3072 (32×32×3) | 2.80 bits/dim (SOTA) |
| Enwik8 文本 | 12288 | 0.99 bits/char (SOTA) |
| 古典音乐生成 | ~65000 | 生成连贯长音频片段 |

## 六、为什么重要？

1. **首次证明** Transformer 可以高效处理超长序列
2. **启发了后续工作**: Longformer, BigBird, Linear Attention 等
3. **多模态能力**: 统一处理文本、图像、音频
4. **实用性**: 自定义 CUDA 内核实现了理论加速

## 七、与后续工作的关系

```
Sparse Transformer (2019)
    ├── Longformer (2020) — 滑动窗口 + 全局token
    ├── BigBird (2020) — 随机 + 局部 + 全局注意力
    ├── Linear Attention — 线性复杂度近似
    └── FlashAttention (2022) — 从IO角度优化标准注意力
```
