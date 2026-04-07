---
id: h2o
year: 2023
tier: advanced
tags: [kv-cache, eviction, memory, long-context, heavy-hitter, greedy]
requires: [paged-attention, prefill-decode]
summary: "Heavy Hitter Oracle keeps top-k attended tokens, achieving 20× KV cache compression."
equations:
  - "score(t) = Σᵢ₌ₜ^T αᵢₜ  (accumulated attention mass)"
  - "KV_kept = HH_set ∪ recent_window"
complexity: {time: "O(1) eviction decision per new token", memory: "O(k) KV slots, k << n"}
paper: {title: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models", authors: "Zhang et al.", year: 2023}
viz: 07-h2o.html
---

## One-liner
H2O evicts KV cache entries greedily by keeping only the "heavy hitter" tokens with the highest accumulated attention scores plus a recent window.

## Key equations
```
# Attention score accumulation: score for token t at decode step T
score(t) = Σᵢ₌ₜ^T  αᵢₜ       # sum of attention weights received by t
                               # αᵢₜ = softmax( qᵢ · kₜ / √d_k )

# H2O eviction policy (budget = k total KV slots):
HH_set    = top-(k - w) tokens by accumulated score
recent_w  = last w tokens (recency window)
KV_kept   = HH_set ∪ recent_w   # |KV_kept| = k

# On each new decode step:
1. Compute attention over current KV_kept
2. Update scores for kept tokens
3. If |KV_kept| > k: evict token with lowest score not in recent_w

# Compression ratio
ratio = n / k    # e.g., n=2048, k=100 → 20× compression

# Greedy optimality claim
Theorem (H2O): greedy score-based eviction minimizes the L1 error
of the attention output compared to full-cache attention, under
the assumption of score monotonicity.
```

## Why it matters
For long-context inference, the KV cache can exceed the model weights in GPU memory, limiting batch size to 1 or requiring expensive offloading. H2O shows empirically that attention mass is highly concentrated: a small subset of "heavy hitter" tokens consistently receive most attention across all layers and heads. Keeping only these tokens plus a recency buffer preserves output quality with up to 20× reduction in KV cache size, enabling much larger effective batch sizes or longer contexts within the same memory budget.

## Gotchas
- Accumulated attention scores are computed per head; aggregation across heads (sum, max, mean) affects quality — the paper uses sum.
- The recency window is critical: without it, the model loses track of the immediate context and output degrades sharply even at small compression ratios.
- Score accumulation is approximate during prefill if using chunked prefill — partial scores from early chunks must be correctly merged.
- H2O assumes static eviction policy per request; it does not adapt to query distribution changes mid-sequence.
- At high compression ratios (>20×), degradation on tasks requiring precise recall of early tokens (e.g., passkey retrieval) is significant.

## Code pointer
`FMInference/H2O` → `h2o_hf.py` → `H2OLlamaAttention.forward()` — wraps HuggingFace attention with score tracking and greedy eviction.
