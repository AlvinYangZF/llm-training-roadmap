---
id: turboquant
year: 2026
tier: applied
tags: [quantization, inference, speedup, int4, post-training]
requires: [quantization, prefill-decode]
summary: "PTQ pipeline combining weight + activation quantization for INT4 inference speedup."
equations:
  - "W_q = round(W / s) * s,  s = max(|W|) / (2^(b-1) - 1)"
  - "TTFT ∝ 1/arithmetic_throughput,  TPOT ∝ 1/memory_bandwidth"
complexity: {time: "O(C) calibration, O(1) amortised inference overhead", memory: "O(N/2) vs FP16 for INT4 weights (4x compression)"}
paper: {title: "TurboQuant: Efficient Ultra-Low-Bit Post-Training Quantization for LLM Inference", authors: "TurboQuant Team", year: 2026}
viz: 11-turboquant.html
---

## One-liner
A post-training quantization pipeline that jointly quantizes weights and activations to INT4, using a calibration dataset to minimize output error without any gradient updates.

## Key equations
```
# Per-channel weight quantization
s_w = max(|W_c|) / (2^(b-1) - 1)
W_q = clamp(round(W / s_w), -(2^(b-1)), 2^(b-1) - 1)

# Activation quantization (per-token dynamic)
s_a = max(|X_t|) / (2^(b-1) - 1)          # computed at runtime per token

# Quantization error minimization (calibration objective)
argmin_{s_w} || W·X - dequant(quant(W, s_w)) · X ||_F
         over calibration set X ~ D_cal

# Throughput model
TTFT_speedup ≈ FP16_FLOPS / INT4_FLOPS   ≈ 4×  (compute-bound prefill)
TPOT_speedup ≈ FP16_BW   / INT4_BW       ≈ 2–3× (memory-bound decode)
```

## Why it matters
Inference cost at scale is dominated by memory bandwidth during autoregressive decode (TPOT) and by arithmetic throughput during long-context prefill (TTFT). INT4 weights cut memory footprint 4× vs FP16, directly accelerating decode on bandwidth-limited GPUs. Combining activation quantization lets the matrix multiply stay in INT4 throughout, unlocking INT4 tensor core throughput. PTQ requires only a small calibration set (512–2048 samples), making it deployable without access to full training data or GPU clusters. The key challenge is outlier activations in attention and FFN layers — TurboQuant addresses this with per-token dynamic activation scaling and channel-wise weight reordering.

## Gotchas
- Activation outliers (common in LLaMA-family models at specific channels) cause large INT4 clipping errors; smooth-quant-style channel migration is required.
- INT4 quantization of KV cache is separate from weight/activation quantization — conflating them leads to incorrect latency projections.
- Calibration set distribution matters: out-of-distribution prompts can surface accuracy cliffs post-deployment even if calibration perplexity looks fine.
- INT4 GEMM kernels require specific tile sizes; small batch sizes (decode) may not saturate tensor cores, reducing effective speedup below theoretical 4×.
- Some layers (embedding, LM head, first/last transformer layers) typically must stay in FP16 to preserve output quality.

## Code pointer
`turboquant/quantize.py` → `TurboQuantizer.quantize_model()`
