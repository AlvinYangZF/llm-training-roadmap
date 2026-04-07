---
id: quantization
year: 2023
tier: applied
tags: [quantization, int8, int4, gptq, awq, memory, post-training, bnb]
requires: [training-loss, linear-algebra]
summary: "Reduce weight precision to INT8/INT4 for 2–4× memory reduction with minimal accuracy loss."
equations:
  - "Q(w) = round(w / scale) * scale"
  - "scale = max(|W|) / (2^(b-1) - 1)"
complexity:
  time: "O(n²) Hessian per layer (GPTQ); O(1) dequant at inference"
  memory: "O(params / 8) bytes at INT4 vs O(params · 2) at BF16"
paper:
  title: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
  authors: "Frantar, Ashkboos, Hoefler, Alistarh"
  year: 2022
viz: 26-quantization.html
---

## One-liner
Post-training quantization maps FP32/BF16 weights to low-bit integers, cutting memory by 2–4x with calibration-based error minimisation.

## Key equations
```
absmax quantization:
  scale    = max(|W|) / (2^(b-1) - 1)
  Q(w)     = round(w / scale) * scale
  error    = ||W - Q(W)||_F

GPTQ (layer-wise second-order):
  min_Q  ||WX - Q(W)X||^2_F
  solved via Hessian H = 2*XX^T, column-wise greedy update

AWQ (activation-aware):
  scale s_i = (mean|X_i|)^alpha / (mean|W_i|)^(1-alpha)
  Q( W / s ) * s     # protect salient channels via per-channel scaling
```

## Why it matters
A 70B parameter model in BF16 requires ~140 GB VRAM — beyond single-GPU capacity. 4-bit quantization brings this to ~35 GB, fitting on a single A100 80GB. GPTQ uses second-order (Hessian) information to redistribute quantization error across weights within each linear layer, achieving INT4 quality comparable to BF16 baselines. AWQ exploits the observation that 1% of weight channels are activation-sensitive: protecting those channels and scaling the rest recovers most accuracy. bitsandbytes uses block-wise absmax (separate scale per 64-weight block) for simpler but effective INT8 quantization.

## Gotchas
- Activations are almost always kept in BF16 at inference; only weights are quantized in PTQ — quantizing activations (W8A8) requires careful calibration data.
- GPTQ calibration requires a small dataset (~128 samples); calibration set distribution matters — mismatch degrades results.
- AWQ and GPTQ quantised models are not interchangeable: different packing formats, different dequantisation kernels.
- 4-bit models on GPU require custom CUDA kernels (exllama, marlin, AWQ kernels) for fast dequant during matmul; naive dequant-then-matmul is slow.
- Double quantisation (QLoRA): quantise the quantisation constants themselves to save an additional ~0.5 bits/param.

## Code pointer
`auto_gptq/modeling/_base.py` → `BaseGPTQForCausalLM.quantize()` / `bitsandbytes/nn/modules.py` → `Linear4bit.forward()`
