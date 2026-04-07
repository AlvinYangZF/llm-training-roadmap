---
id: prefill-decode
year: 2022
tier: core
tags: [inference, kv-cache, memory, throughput, phases]
requires: [transformer, gpt2]
summary: "Two-phase LLM inference: parallel prompt processing then sequential token generation."
equations:
  - "KV_cache[l][t] = (Wₖ hₜˡ, Wᵥ hₜˡ)"
  - "TTFT = time to first token (prefill latency)"
  - "TPOT = time per output token (decode latency)"
complexity: {time: "Prefill O(n²d) once; Decode O(nd) per token", memory: "O(n·d·L·2) for KV cache"}
paper: {title: "Efficiently Scaling Transformer Inference", authors: "Pope et al.", year: 2022}
viz: 04-prefill-decode.html
---

## One-liner
Prefill processes the entire prompt in one parallel forward pass; decode generates output tokens one at a time, reusing the cached K and V tensors.

## Key equations
```
# KV cache: store K and V for each layer l, each token position t
KV_cache[l][t] = ( Wₖ · hₜˡ,  Wᵥ · hₜˡ )

# Decode step t+1 only computes new K,V for position t+1
# then attends over full KV_cache[0..t+1]
attn_out = softmax( qₜ₊₁ · Kᵀ_cache / √d_k ) · V_cache

# Memory cost of KV cache
KV_bytes = 2 · n_layers · n_heads · d_head · seq_len · sizeof(dtype)
# e.g., LLaMA-7B, fp16, seq=4096: 2·32·32·128·4096·2 ≈ 2 GB

# Roofline: arithmetic intensity threshold
# Compute-bound if FLOPs/byte > hardware ridge point
# Prefill: AI ≈ seq_len/2  → typically compute-bound
# Decode:  AI ≈ 1          → memory-bandwidth-bound

# Latency metrics
TTFT  = time from request arrival to first output token  (prefill-dominated)
TPOT  = time per additional output token                 (decode-dominated)
E2E   = TTFT + (output_len - 1) × TPOT
```

## Why it matters
Understanding the two phases is essential for LLM serving optimization. Prefill is compute-bound (GPU utilization is high); decode is memory-bandwidth-bound (utilization is low because only one token's activations are computed per step). The KV cache trades memory for avoiding redundant K/V recomputation — without it, each decode step would cost O(t²) instead of O(t). TTFT and TPOT are the two independent SLOs that serving systems optimize separately.

## Gotchas
- KV cache grows linearly with sequence length; for long contexts it can easily exceed model weights in GPU memory.
- Continuous batching (iteration-level scheduling) keeps GPUs busy by mixing prefill and decode requests, but naively mixing them in one batch causes the decode throughput to degrade to prefill latency.
- fp16 KV cache halves memory vs fp32 but introduces small accuracy loss; quantized KV cache (int8/int4) is an active area.
- TTFT can be dominated by queue wait time in production, not just prefill compute — always measure end-to-end including scheduling.
- Decode is not purely sequential: speculative decoding and parallel decoding (Medusa, Jacobi) break the one-token-per-step constraint.

## Code pointer
`vllm/worker/model_runner.py` → `ModelRunner.execute_model()` — dispatches prefill vs decode paths and manages the KV cache.
