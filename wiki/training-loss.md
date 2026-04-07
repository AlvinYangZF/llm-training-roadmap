---
id: training-loss
year: 2017
tier: core
tags: [training, loss, cross-entropy, perplexity, backprop, adam, gradient]
requires: [transformer, tokenization]
summary: "Next-token cross-entropy loss, backprop, Adam, and cosine LR schedule."
equations:
  - "L = -Σ_t log p(x_t | x_{<t})"
  - "PPL = exp(L)"
  - "Adam: m = β₁m + (1-β₁)g, v = β₂v + (1-β₂)g²"
complexity:
  time: "O(T · d²) per forward pass, same for backward"
  memory: "O(params · 12) bytes for Adam states in FP32"
paper:
  title: "Attention Is All You Need"
  authors: "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin"
  year: 2017
viz: 25-training-loss.html
---

## One-liner
LLMs minimise next-token cross-entropy via Adam with cosine LR decay and gradient clipping; perplexity is the exponential of the average loss.

## Key equations
```
loss (per token):  L = -(1/T) Σ_t log p(x_t | x_1, ..., x_{t-1})

perplexity:        PPL = exp(L)

Adam update:
  m_t = β₁ m_{t-1} + (1 - β₁) g_t          # 1st moment (momentum)
  v_t = β₂ v_{t-1} + (1 - β₂) g_t²         # 2nd moment (RMS)
  m̂_t = m_t / (1 - β₁ᵗ)                    # bias correction
  v̂_t = v_t / (1 - β₂ᵗ)
  θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)

cosine LR:  lr(t) = lr_min + 0.5·(lr_max - lr_min)·(1 + cos(π·t/T_max))
```

## Why it matters
Next-token prediction is a self-supervised objective: any raw text corpus provides supervision with no human labels. The cross-entropy loss directly maximises log-likelihood of the training distribution. Perplexity is the standard intrinsic evaluation metric — it measures how surprised the model is by held-out data on average. Adam adapts per-parameter learning rates using gradient history, which is critical for transformers where gradient magnitudes vary widely across layers and heads. Gradient clipping (‖g‖ > 1.0 → g ← g/‖g‖) prevents catastrophic loss spikes from exploding gradients, especially during early training.

## Gotchas
- Adam stores 2 extra copies of all parameters (m and v) in FP32, tripling memory vs inference; use AdamW (adds weight decay) not vanilla Adam.
- Warmup steps (typically 1–4% of total steps) are required: jumping to peak LR from random init causes early divergence.
- Cosine decay should anneal to lr_min ≈ lr_max/10, not 0; too-low final LR wastes training compute.
- Gradient accumulation emulates larger batch sizes but changes effective batch norm statistics if any are present.
- Loss spikes (sudden 2–5× increase) during training often indicate a corrupted batch or a data quality issue, not a bug in the optimizer.

## Code pointer
`transformers/trainer.py` → `Trainer.compute_loss()` / `torch/optim/adamw.py` → `AdamW.step()`
