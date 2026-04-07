---
id: transformer
year: 2017
tier: core
tags: [attention, architecture, foundation, self-attention, multi-head]
requires: [linear-algebra, embeddings]
summary: "Self-attention + FFN blocks that replaced RNNs as the universal sequence model."
equations:
  - "Attention(Q,K,V) = softmax(QKᵀ / √d_k) V"
  - "MultiHead(Q,K,V) = Concat(head₁,...,headₕ) Wᴼ"
  - "FFN(x) = max(0, xW₁ + b₁)W₂ + b₂"
  - "LayerNorm(x) = (x − μ) / σ · γ + β"
complexity: {time: "O(n² d) per layer", memory: "O(n²) for attention matrix"}
paper: {title: "Attention Is All You Need", authors: "Vaswani et al.", year: 2017}
viz: 01-transformer-basics.html
---

## One-liner
The transformer replaces recurrence with scaled dot-product attention, enabling full parallel training over sequences.

## Key equations
```
# Scaled dot-product attention
Attention(Q, K, V) = softmax( Q Kᵀ / sqrt(d_k) ) · V

# Multi-head attention (h heads, each projects to d_k = d_model/h)
headᵢ = Attention(Q Wᵢᵠ, K Wᵢᴷ, V Wᵢᵛ)
MultiHead(Q,K,V) = Concat(head₁,...,headₕ) Wᴼ

# Position-wise FFN (expand 4×, then contract)
FFN(x) = max(0, x W₁ + b₁) W₂ + b₂

# Pre-LayerNorm residual stream (modern variant)
x ← x + Sublayer(LayerNorm(x))

# Sinusoidal positional encoding
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

## Why it matters
Parallelism over sequence positions enables training on orders-of-magnitude more data than RNNs allowed. The attention mechanism learns which positions are relevant to each other without any inductive positional bias, making it a general-purpose in-context lookup table. Residual connections let gradients flow cleanly through many layers. The same architecture, scaled up, became GPT, BERT, T5, and every modern LLM.

## Gotchas
- The √d_k scaling prevents softmax from entering saturation (near-zero gradients) when d_k is large; omitting it tanks training.
- Original "Attention Is All You Need" uses Post-LN (norm after residual); most modern LLMs use Pre-LN for stability.
- Positional encoding must be added to embeddings *before* the first layer; it encodes absolute position, not relative.
- Causal (autoregressive) models add a triangular mask to the attention logits — without it the model cheats by attending to future tokens.
- Multi-head attention uses h separate weight matrices; a single large projection is **not** equivalent because it lacks the independent subspace inductive bias.

## Code pointer
`transformers/models/gpt2/modeling_gpt2.py` → `GPT2Attention.forward()` — reference implementation of masked multi-head attention with KV caching.
