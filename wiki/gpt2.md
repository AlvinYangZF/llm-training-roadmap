---
id: gpt2
year: 2019
tier: core
tags: [architecture, autoregressive, decoder-only, language-model, causal]
requires: [transformer]
summary: "Decoder-only causal transformer; foundation of every GPT-style language model."
equations:
  - "P(xₜ | x<t) = softmax(h_L Wᵀ_e)[xₜ]"
  - "L = -Σₜ log P(xₜ | x₁,...,xₜ₋₁)"
complexity: {time: "O(n² d) prefill, O(nd) per decode step", memory: "O(n·d·L) for KV cache (L layers)"}
paper: {title: "Language Models are Unsupervised Multitask Learners", authors: "Radford et al.", year: 2019}
viz: 02-gpt2-architecture.html
---

## One-liner
GPT-2 is a stack of causally-masked transformer decoder blocks trained on next-token prediction — no encoder, no cross-attention.

## Key equations
```
# Token + position embedding
h₀ = W_e[xₜ] + W_p[t]        # W_e ∈ ℝ^(V×d), W_p ∈ ℝ^(n_ctx×d)

# Causal mask: upper triangle set to -∞ before softmax
mask[i,j] = 0 if j ≤ i else -∞

# Output logits (weight tying: reuse embedding matrix)
logits = h_L · W_eᵀ          # shape: (n, V)

# Autoregressive loss (cross-entropy)
L = -1/T · Σₜ log P(xₜ | x<t)

# GPT-2 model sizes
GPT-2 Small:  12 layers, 12 heads, d=768,  117M params
GPT-2 Medium: 24 layers, 16 heads, d=1024, 345M params
GPT-2 Large:  36 layers, 20 heads, d=1280, 774M params
GPT-2 XL:     48 layers, 25 heads, d=1600, 1.5B params
```

## Why it matters
GPT-2 demonstrated that a single architecture trained on raw web text learns grammar, facts, and basic reasoning with no task-specific supervision. It established the decoder-only paradigm that all subsequent GPT models (GPT-3, GPT-4, LLaMA, Mistral) inherit. Weight tying between the input embedding and the output projection matrix reduces parameters and improves perplexity. The causal mask is the key architectural decision that makes autoregressive generation exact and efficient.

## Gotchas
- Weight tying means `model.lm_head.weight is model.transformer.wte.weight` — modifying one modifies both.
- The context window is hard-coded (1024 tokens for GPT-2); unlike RoPE-based models, standard learned positional embeddings do not extrapolate beyond training length.
- BPE tokenizer has a fixed vocabulary (50,257 tokens); unknown bytes are encoded via a byte-level fallback, so GPT-2 never produces UNK.
- Autoregressive generation is sequential: token t+1 depends on token t, so you cannot parallelize the decode loop itself.
- Pre-LN was adopted in GPT-2 (vs original transformer's Post-LN), which stabilizes training at depth but shifts residual stream statistics.

## Code pointer
`transformers/models/gpt2/modeling_gpt2.py` → `GPT2LMHeadModel.forward()` — full forward pass including weight-tied logits.
