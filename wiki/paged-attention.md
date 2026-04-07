---
id: paged-attention
year: 2023
tier: advanced
tags: [kv-cache, memory, serving, vllm, paging, fragmentation]
requires: [prefill-decode]
summary: "Virtual memory paging for KV cache eliminates fragmentation in LLM serving."
equations:
  - "Utilization = used_slots / total_slots → ~100% vs ~74% baseline"
  - "CoW: page ref_count > 1 → copy on write for fork"
complexity: {time: "O(1) block alloc/free per step", memory: "O(seq_len / block_size) block table entries"}
paper: {title: "Efficient Memory Management for Large Language Model Serving with PagedAttention", authors: "Kwon et al.", year: 2023}
viz: 06-paged-attention.html
---

## One-liner
PagedAttention maps a logical KV cache to non-contiguous physical memory blocks, eliminating internal and external fragmentation for LLM serving.

## Key equations
```
# Block table: logical block number → physical block number
block_table[seq_id][logical_block] = physical_block_id

# Each physical block holds block_size token KV slots
block_size = B   # e.g., B = 16 tokens

# Maximum possible fragmentation per sequence
waste ≤ (block_size - 1) tokens per sequence   # internal fragmentation only

# Memory utilization comparison
Baseline (contiguous reservation):  ~74% utilization (worst case)
PagedAttention (dynamic blocks):    ~96-100% utilization

# Copy-on-Write for parallel sampling (beam search / top-k)
fork(seq_A → seq_B):
    share all blocks (increment ref_count)
    on write to shared block: copy → new block, decrement ref_count

# Throughput gain from packing
# More sequences fit in GPU memory → higher batch size → better GPU utilization
throughput ∝ effective_batch_size ∝ 1 / fragmentation_overhead
```

## Why it matters
Before PagedAttention, LLM serving frameworks pre-allocated contiguous KV cache buffers at the maximum sequence length, wasting 20-40% of GPU memory to fragmentation. PagedAttention borrows OS virtual memory ideas: fixed-size blocks are allocated on demand as a sequence grows, and freed immediately when done. This dramatically increases the number of concurrent sequences per GPU, improving throughput by 2-4× on real workloads. The copy-on-write mechanism makes parallel sampling (beam search, best-of-N) nearly free in memory.

## Gotchas
- Block size is a tuning parameter: small blocks reduce fragmentation but increase block table overhead and hurt memory locality; 16-32 tokens is typical.
- The block table lookup adds an indirection per attention step; this is cheap on CPU but requires careful CUDA implementation to avoid per-token overhead.
- Copy-on-write works only if the engine tracks reference counts correctly — a bug silently corrupts two sequences sharing a block.
- PagedAttention assumes fixed token KV size; variable d_head (e.g., MLA in DeepSeek) requires custom block layout.
- Prefix caching (sharing KV blocks for common prefixes across requests) is a natural extension but requires immutable blocks and a hash-based lookup.

## Code pointer
`vllm/core/block_manager.py` → `BlockSpaceManager.allocate()` and `BlockSpaceManager.fork()` — block allocation and CoW logic.
