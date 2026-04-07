---
id: cag-vs-rag
year: 2025
tier: applied
tags: [rag, cag, long-context, cache, kv-cache, tradeoffs, context-window]
requires: [rag-pipeline, paged-attention]
summary: "CAG preloads KB into KV cache; RAG retrieves at query time — each wins in different regimes."
equations:
  - "CAG latency = T_prefill(once) + T_decode(query)"
  - "RAG latency = T_retrieve + T_prefill(chunks) + T_decode(query)"
complexity:
  time: "CAG: O(1) retrieval, O(n²) prefill once; RAG: O(log n) ANN per query"
  memory: "CAG: O(n·L·d) KV cache pinned; RAG: O(k·L·d) per query"
paper:
  title: "Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks"
  authors: "Chan, Lam, Yang, Zhu"
  year: 2025
viz: 22-cag-vs-rag.html
---

## One-liner
Cache-Augmented Generation (CAG) preloads the entire knowledge base as a reusable KV cache, eliminating retrieval latency when the KB is small and queries are stable.

## Key equations
```
CAG:  response = LLM(query | KV_cache(KB))
      KV_cache built once, reused for all queries

RAG:  retrieved = ANN_search(embed(query), index)
      response  = LLM(query | retrieved_chunks)

KV cache size ≈ 2 · n_layers · n_heads · d_head · seq_len  (bytes = × dtype_size)
```

## Why it matters
Modern long-context models (Gemini 1.5 Pro: 1M tokens; Claude: 200K) make it feasible to fit entire domain corpora into a single context window. If the same KB answers many queries, the quadratic prefill cost is amortised — paid once, not per query. CAG sidesteps retrieval failure modes: no chunking decisions, no ANN recall gaps, no relevance scoring errors. However, KV cache for 128K tokens at BF16 with a 70B model can exceed 50GB of GPU VRAM, and the approach collapses when the KB grows beyond context capacity or changes frequently.

## Gotchas
- KV cache must be regenerated on any KB update — no incremental patching in current implementations.
- Attention over very long cached contexts degrades for some models; performance varies by architecture (RoPE scaling quality matters).
- GPU VRAM for KV cache scales linearly with sequence length: measure before committing to CAG.
- CAG conflates all KB content into context; irrelevant passages can increase hallucination risk compared to targeted RAG retrieval.
- Radix attention / prefix caching (vLLM, SGLang) is required to share the cached prefix across concurrent requests; without it, VRAM is duplicated per request.

## Code pointer
`sglang/python/sglang/srt/managers/scheduler.py` → `PrefixCache.match_prefix()`
