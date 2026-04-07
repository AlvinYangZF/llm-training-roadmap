---
id: pd-separation
year: 2025
tier: advanced
tags: [serving, disaggregation, prefill, decode, distributed, gpu-utilization]
requires: [prefill-decode, paged-attention]
summary: "Separate prefill and decode to dedicated GPU pools for optimal utilisation."
equations:
  - "GPU_util(prefill) = compute-bound; GPU_util(decode) = memory-bandwidth-bound"
  - "optimal: run prefill on high-FLOP GPUs, decode on high-memory-BW GPUs"
complexity: {time: "prefill O(n²) compute, decode O(n) per step", memory: "KV transfer cost = n_layers × 2 × d_model × n_heads × seq_len × sizeof(dtype)"}
paper: {title: "Splitwise: Efficient Generative LLM Inference Using Phase Splitting", authors: "Patel et al.", year: 2024}
viz: 09-pd-separation.html
---

## One-liner
Prefill/decode disaggregation routes the compute-heavy prefill phase to high-FLOP GPU pools and memory-bandwidth-bound decode to separate pools, eliminating interference and boosting overall throughput.

## Key equations
```
# The fundamental bottleneck mismatch:
prefill:  arithmetic_intensity = FLOPs / bytes = O(seq_len)  → compute-bound
decode:   arithmetic_intensity = FLOPs / bytes = O(1)        → memory-BW-bound
# Running both on same GPU means each phase underutilises the other's bottleneck

# KV cache transfer on disaggregation boundary:
transfer_bytes = n_layers × 2 × d_model × seq_len × sizeof(fp16)
# e.g. LLaMA-70B (80 layers, d=8192) at seq=512: ~10 GB — must go over NVLink/RDMA

# Throughput model (simplified):
throughput_batched = batch_size / (t_prefill/n_prefill_gpus + t_transfer + t_decode/n_decode_gpus)
# optimal split: n_prefill_gpus / n_decode_gpus = t_prefill / t_decode

# GPU fleet allocation:
if TTFT (time-to-first-token) SLA is tight → allocate more prefill GPUs
if TBT (time-between-tokens) SLA is tight  → allocate more decode GPUs
```

## Why it matters
In a co-located prefill+decode setup, long prefill requests stall in-flight decode batches (TTFT spikes) while bursty decode load wastes prefill GPU cycles. Disaggregation allows each pool to be independently scaled, batched, and hardware-optimised — prefill clusters can use H100 SXM for raw FLOP density while decode clusters can use memory-bandwidth-optimised chips. Mooncake and Deepseek-V3 serving both adopt this architecture. At datacenter scale, it yields 40-60% better GPU utilisation over co-located serving.

## Gotchas
- KV transfer bandwidth is the new bottleneck: 10–40 GB per prefilled sequence over PCIe will nullify latency savings; requires NVLink or RDMA (InfiniBand/RoCE).
- Load imbalance between pools degrades the weaker pool to a bottleneck; dynamic rerouting (Mooncake scheduler) or elastic scaling is needed.
- Short prompts (< 128 tokens) are prefill-trivial; disaggregation overhead dominates for chatbot-style short requests — only beneficial at long-context or high-throughput regimes.
- KV cache compression (quantization, pruning) reduces transfer cost and is almost mandatory for disaggregation to be practical at 70B+ model sizes.
- Stateful decode contexts (multi-turn chat) must track which decode GPU owns which session's KV state; naïve load balancing breaks session affinity.

## Code pointer
`vllm-project/vllm` → `vllm/core/scheduler.py` — scheduler logic for prefill vs decode queue separation; disaggregated mode enabled via `--disaggregated-prefill` flag in v0.5+.
