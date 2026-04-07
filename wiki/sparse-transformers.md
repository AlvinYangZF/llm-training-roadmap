---
id: sparse-transformers
year: 2019
tier: advanced
tags: [attention, sparsity, efficiency, long-context, strided]
requires: [transformer]
summary: "Structured sparse attention cuts O(n²) to O(n√n) for long sequences."
equations:
  - "A_sparse(i) = A_local(i) ∪ A_strided(i)"
  - "Complexity: O(n · √n) vs O(n²) dense"
complexity: {time: "O(n√n) with block-sparse patterns", memory: "O(n√n) attention storage"}
paper: {title: "Generating Long Sequences with Sparse Transformers", authors: "Child et al.", year: 2019}
viz: 03-sparse-transformers.html
---

## One-liner
Sparse Transformers replace full O(n²) attention with structured sparse patterns, enabling transformers to model sequences of tens of thousands of tokens.

## Key equations
```
# Stride r = √n; two attention head types:

# Local head: attend to previous l tokens (sliding window)
A_local(i) = { j : i - l ≤ j ≤ i }

# Strided head: attend every r-th token (long-range summary)
A_strided(i) = { j : j ≡ i (mod r) }

# Combined sparse set for factorized attention
A_sparse(i) = A_local(i) ∪ A_strided(i)

# Attention complexity comparison
Dense:  O(n²)         (all pairs)
Sparse: O(n · √n)     (each token attends to ≈ 2√n positions)

# Block-sparse formulation: partition sequence into blocks of size r
# Local block: attend within block
# Strided block: attend to every r-th block
```

## Why it matters
Full attention is quadratic in sequence length, making n > 4096 prohibitively expensive in both compute and memory. Sparse Transformers showed that restricting each token to attend to O(√n) positions via local + strided patterns preserves most of the modeling capacity for structured data (images, music, text). The factorized patterns ensure every token can reach every other token in at most two hops, maintaining the long-range connectivity that makes transformers powerful.

## Gotchas
- Stride r must equal √n for the O(n√n) complexity claim; a fixed r gives O(n²/r) which is still quadratic unless r grows with n.
- Sparse attention requires custom CUDA kernels (block-sparse matmul) to realize the theoretical speedup — naive masked dense attention saves no FLOPs.
- Two-head factorization (local + strided) is not universal; it works well for sequences with local structure (text, audio) but degrades for tasks requiring arbitrary long-range dependencies.
- Sparse patterns complicate KV caching during inference: the irregular access pattern breaks simple sequential cache indexing.
- Flash Attention (2022) largely superseded sparse attention for moderate lengths by making dense attention IO-efficient; sparse patterns remain relevant beyond ~16K tokens.

## Code pointer
`openai/sparse_attention` (GitHub) → `blocksparse_transformer.py` → `BlocksparseTransformerSelfAttention.forward()` — reference block-sparse attention kernel integration.
