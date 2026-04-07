---
id: linear-algebra
year: 1900
tier: core
tags: [math, vectors, matrices, dot-product, foundations]
requires: []
summary: "Vectors, matrices, dot products — math backbone of every LLM operation."
equations:
  - "a · b = Σ aᵢbᵢ = ‖a‖‖b‖cosθ"
  - "(AB)ᵢⱼ = Σₖ AᵢₖBₖⱼ"
  - "‖x‖₂ = √(Σ xᵢ²)"
  - "Av = λv"
complexity: {time: "O(n²) matmul naive, O(n^2.37) Strassen-like", memory: "O(n²) for n×n matrix"}
paper: {title: "", authors: "", year: 1900}
viz: 00-linear-algebra.html
---

## One-liner
Linear algebra provides the vector and matrix operations that underlie every forward pass in a transformer.

## Key equations
```
# Dot product
a · b = Σ aᵢbᵢ = ‖a‖‖b‖cosθ

# Matrix multiply: C = AB, C is (m×p), A is (m×n), B is (n×p)
Cᵢⱼ = Σₖ AᵢₖBₖⱼ

# L2 norm
‖x‖₂ = sqrt(Σ xᵢ²)

# Eigenvalue decomposition
A v = λ v   →   A = Q Λ Qᵀ  (for symmetric A)

# Transpose
(AB)ᵀ = BᵀAᵀ
```

## Why it matters
Every token embedding is a vector in ℝᵈ; attention is a scaled dot product between vectors; Q, K, V projections are matrix multiplies. Layer norm, FFN, and output projection are all linear maps with bias. Positional encodings use sinusoidal functions but the encoding is added as a vector to the embedding. Understanding matrix shapes and how dimensions flow through a forward pass is prerequisite knowledge for debugging every LLM component.

## Gotchas
- Matrix multiply is **not** commutative: AB ≠ BA in general; shape mismatches are the #1 runtime error.
- Dot product measures cosine similarity only when both vectors are L2-normalized; raw dot products conflate magnitude and angle.
- Eigendecomposition assumes a square matrix; most weight matrices are rectangular — use SVD instead (W = UΣVᵀ).
- Broadcasting silently promotes scalars/vectors to match batch dimensions; a shape bug can propagate invisibly.
- In transformers, d_model is split into d_head × n_heads; forgetting the split leads to wrong attention dimensionality.

## Code pointer
`torch.nn.Linear` → `F.linear(input, weight, bias)` — wraps a matmul + optional bias add.
