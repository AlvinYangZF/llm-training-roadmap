---
id: flash-attention
year: 2022
tier: advanced
tags: [attention, compute, memory-io, cuda, kernel, io-aware]
requires: [transformer, prefill-decode]
summary: "IO-aware tiled attention avoids materializing the n×n matrix, 2-4× faster."
equations:
  - "Standard attention memory: O(n²)"
  - "FlashAttention memory: O(n)"
  - "Online softmax: mᵢ = max(mᵢ₋₁, rowmax(Sᵢ))"
complexity: {time: "O(n²d) FLOPs same as standard; fewer HBM reads/writes", memory: "O(n) activations stored"}
paper: {title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", authors: "Dao et al.", year: 2022}
viz: 05-flash-attention.html
---

## One-liner
FlashAttention tiles the attention computation to fit in SRAM, avoiding the round-trip to HBM for the full n×n attention score matrix.

## Key equations
```
# Standard attention (materializes n×n S matrix in HBM)
S = Q Kᵀ / √d_k          # n×n, stored in HBM
P = softmax(S)            # n×n, stored in HBM
O = P V                   # n×d, stored in HBM

# FlashAttention: process in tiles of size B_r × B_c (fit in SRAM)
# Online softmax — update running max and normalizer as tiles arrive:
for each tile i of Q, for each tile j of K,V:
    Sᵢⱼ  = Qᵢ Kⱼᵀ / √d_k
    m̃ᵢⱼ  = rowmax(Sᵢⱼ)
    mᵢ   = max(mᵢ_prev, m̃ᵢⱼ)            # update running max
    Pᵢⱼ  = exp(Sᵢⱼ - mᵢ)
    lᵢ   = exp(mᵢ_prev - mᵢ) · lᵢ_prev + rowsum(Pᵢⱼ)  # running sum
    Oᵢ   = diag(exp(mᵢ_prev - mᵢ))⁻¹ · Oᵢ_prev + Pᵢⱼ · Vⱼ
# Final: Oᵢ /= lᵢ

# Memory comparison
Standard:  O(n²) HBM for S and P
Flash:     O(n)  HBM — only O and softmax stats written back

# HBM access comparison (n=4096, d=64)
Standard:  ~2.4 GB HBM reads/writes per attention layer
Flash:     ~0.6 GB  (≈4× reduction in IO)

# FlashAttention-2 additions
- Better work partitioning across warps
- Sequence parallelism: split Q across workers, broadcast K,V
- Supports GQA (grouped-query attention)
```

## Why it matters
Modern GPU throughput is bottlenecked by HBM bandwidth, not raw FLOPs. Standard attention spends most of its time reading/writing the n×n attention matrix to HBM; FlashAttention fuses the entire Q K V O computation into one kernel that stays in fast SRAM. The result is exact attention (not approximate) with 2-4× wall-clock speedup and O(n) memory, enabling longer sequences that were previously infeasible. FlashAttention-2 became the de-facto attention backend in PyTorch 2.0+ (`F.scaled_dot_product_attention`).

## Gotchas
- FlashAttention computes exact attention — it is not an approximation; numerical results match standard attention up to floating-point rounding.
- Tile size B_r × B_c must fit in SRAM (≈192 KB on A100 per SM); if d_head is too large (e.g., 256), block sizes shrink and efficiency degrades.
- The custom CUDA kernel requires specific GPU architectures; older GPUs (pre-Ampere) see less benefit because SRAM is smaller.
- Backward pass requires re-materializing tiles on the fly (no saved attention matrix), increasing backward FLOPs by a small constant factor.
- FA2's sequence parallelism splits the sequence across GPUs but requires an all-gather for K and V — introduces communication overhead at large scales.

## Code pointer
`Dao-AILab/flash-attention` → `flash_attn/flash_attn_interface.py` → `flash_attn_func()` — main entry point; PyTorch 2.0 wraps this via `torch.nn.functional.scaled_dot_product_attention`.
