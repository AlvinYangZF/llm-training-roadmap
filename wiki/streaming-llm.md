---
id: streaming-llm
year: 2024
tier: advanced
tags: [kv-cache, streaming, attention-sink, infinite-context, sliding-window]
requires: [h2o, prefill-decode]
summary: "Attention sinks + sliding window KV cache enable infinite-length text generation."
equations:
  - "KV_kept = sink_tokens ∪ sliding_window[-w:]"
  - "sink_tokens = first k tokens (k=4 typically)"
complexity: {time: "O(1) per decode step (constant window)", memory: "O(sink + window) = O(constant)"}
paper: {title: "Efficient Streaming Language Models with Attention Sinks", authors: "Xiao et al.", year: 2024}
viz: 08-streaming-llm.html
---

## One-liner
StreamingLLM keeps a tiny fixed set of "attention sink" tokens plus a sliding recency window, enabling truly infinite-length generation without recomputing or reloading the full KV cache.

## Key equations
```
# Attention sink observation:
# First few tokens receive disproportionate attention regardless of content
# Removing them causes attention score distribution to collapse → quality collapse

# StreamingLLM KV retention policy:
KV_kept = sink_tokens ∪ sliding_window
sink_tokens    = KV for positions 0..s-1  (s=4)
sliding_window = KV for last w tokens     (w=1024 typical)
|KV_kept| = s + w = constant

# Positional encoding fix (critical for RoPE/ALiBi models):
# Re-index kept tokens by their original position, not dense-packed index
# Without this: position 0,1,2,3,1021,1022,1023,1024 → must use original pos IDs
# Most implementations store (position_id, key, value) tuples

# Memory is strictly bounded:
GPU_mem(KV) = (s + w) × n_layers × 2 × d_head × n_heads × sizeof(dtype)
e.g. s=4, w=1024, 32-layer 7B model @ fp16 ≈ 256 MB (vs unbounded without eviction)
```

## Why it matters
Before StreamingLLM, any generation longer than the training context length caused either OOM (full KV cache grows linearly) or quality collapse (when KV eviction removed attention sinks). The key insight is that attention sinks are positional artifacts: models learn to dump excess attention probability on the first token because it's always visible. Keeping just 4 sink tokens restores the full attention distribution even with aggressive eviction of intermediate tokens. This enables deployment scenarios like infinite document summarization, long-running chat agents, and streaming transcription that were previously impractical.

## Gotchas
- Attention sinks are a model artifact — models must be trained with positional stability for sinks to form; some models (e.g. ALiBi-based) exhibit weaker sinks.
- The sliding window does NOT give the model access to tokens evicted from the window — content is truly lost; this is not equivalent to full-context attention.
- Re-indexing position IDs for kept tokens is mandatory for RoPE-based models. Skipping this causes phase mismatches in rotary embeddings and silent quality degradation.
- Adding `[sink]` tokens explicitly during fine-tuning (SinkAttention) gives stronger sinks and higher quality at low window sizes, but requires re-training.
- StreamingLLM is for fixed-length generation quality, not for exact recall. For tasks requiring precise memory of early context, use retrieval instead.

## Code pointer
`mit-han-lab/streaming-llm` → `streaming_llm/enable_streaming_llm.py` → `enable_streaming_llm()` — patches a HuggingFace model in-place to use sink+window KV retention.
