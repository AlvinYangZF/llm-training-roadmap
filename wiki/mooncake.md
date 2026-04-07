---
id: mooncake
year: 2025
tier: advanced
tags: [kv-cache, serving, scheduling, distributed, transfer, reuse]
requires: [pd-separation, paged-attention]
summary: "KV-cache-centric serving that transfers cached KV from prefill to decode nodes."
equations:
  - "reuse_ratio = cache_hits / total_prefill_tokens"
  - "E2E_latency = T_prefill + T_transfer + T_decode"
complexity: {time: "O(L·H·S) per transfer (L layers, H heads, S seq len)", memory: "O(C) where C = total KV cache capacity across cluster"}
paper: {title: "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving", authors: "Ruoyu Qin et al.", year: 2025}
viz: 10-mooncake.html
---

## One-liner
Disaggregated LLM serving where KV cache is the first-class resource: prefill nodes fill it, decode nodes consume it, and a global scheduler maximises reuse.

## Key equations
```
# Transfer cost dominates when cache miss rate is high
T_transfer = KV_bytes / network_bandwidth
           = (2 * L * H * d_head * S * dtype_bytes) / BW

# Effective throughput gain from prefix cache hit
speedup = 1 / (1 - hit_rate + hit_rate * (T_decode_only / T_full_prefill))

# Scheduler objective: maximise hit rate subject to SLO constraints
max  Σ reuse_ratio_i
s.t. TTFT_i ≤ SLO_TTFT,  TBT_i ≤ SLO_TBT
```

## Why it matters
Classic monolithic LLM servers tie prefill and decode to the same GPU, wasting memory bandwidth and causing head-of-line blocking between compute-heavy prefill and memory-bound decode. Mooncake treats the KV cache as a distributed storage layer, routing requests to nodes that already hold matching prefix KV blocks — turning expensive recomputation into cheap network transfers. Prefix caching (radix/hash-based) means system prompts, few-shot examples, and repeated tool definitions are computed once and served many times. By separating the scheduling concern from the compute concern, the system can oversubscribe decode capacity independently of prefill capacity, improving GPU utilisation across heterogeneous workloads.

## Gotchas
- KV transfer bandwidth (NVLink / InfiniBand) must exceed the recompute cost; on slower networks a cache miss is cheaper to recompute than transfer.
- Hash-based prefix matching is exact — a single token difference invalidates the entire suffix; semantic similarity does not help here.
- Cache eviction policy matters: LRU can thrash on long-tail prompts; Mooncake uses a cost-aware policy weighted by recompute cost.
- Decode SLO jitter increases when the transfer pipeline stalls — back-pressure from the decode node must be propagated to the scheduler.
- Memory fragmentation across nodes requires a global compactor; without it, effective cache capacity degrades over time.

## Code pointer
`mooncake/scheduler/kvcache_scheduler.py` → `KVCacheScheduler.schedule_request()`
