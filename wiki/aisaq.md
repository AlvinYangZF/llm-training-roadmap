---
id: aisaq
year: 2024
tier: advanced
tags: [vector-search, quantization, anisotropic, ann, inner-product]
requires: [diskann, quantization]
summary: "Quantize parallel component of vectors with higher precision to preserve dot-product ranking."
equations:
  - "q*(x) = argmin_q  ||x - q||² + λ · (x_⊥ · (q - x))²"
  - "parallel component x_∥ = (x·r̂)r̂,  perpendicular x_⊥ = x - x_∥"
complexity: {time: "O(d) per vector quantization, O(d/w) SIMD dot product with quantized codes", memory: "O(N·b/8) for b-bit codes vs O(N·d·4) for float32"}
paper: {title: "Revisiting the Role of Residual Quantization in ANN Search (ScaNN / AISAQ)", authors: "Ruiqi Guo et al.", year: 2020}
viz: 18-aisaq.html
---

## One-liner
Anisotropic quantization assigns higher precision to the component of a vector parallel to a query direction, minimizing the dot-product error that drives inner-product ranking.

## Key equations
```
# Vector decomposition relative to residual direction r
x_∥ = (x · r̂) r̂             # parallel component (carries most ranking signal)
x_⊥ = x - x_∥                # perpendicular component (contributes less to IP ranking)

# Anisotropic quantization loss (penalise parallel error more)
L_aniso(x, q) = ||x_⊥ - q_⊥||² + λ · ||x_∥ - q_∥||²,   λ >> 1

# Compare to isotropic (standard PQ / scalar quantization):
L_iso(x, q)   = ||x - q||²   # treats all dimensions equally → suboptimal for MIPS

# Inner product approximation after quantization
x · y ≈ q(x) · q(y)          # quantization error in IP: ε ∝ ||x_∥ - q(x)_∥||

# ScaNN overall pipeline
1. Partition corpus into clusters (k-means, ~1000–10000 centroids)
2. Anisotropic quantize residuals within each partition
3. At query time: score all partitions by IP, rescore top-T with quantized codes
4. Exact rescore top-K (retrieve full float32 vectors for final ranking)
```

## Why it matters
Standard PQ and scalar quantization minimize L2 reconstruction error uniformly across all dimensions — but inner product ranking is determined almost entirely by the component of a database vector parallel to the query. Errors in perpendicular components are nearly invisible to MIPS ranking. AISAQ exploits this asymmetry: by allocating more quantization bits to the parallel component, it achieves the same ranking quality as isotropic quantization with far fewer total bits, or equivalently much higher recall at the same bit budget. This is the core insight behind Google ScaNN, which achieves 2–3× throughput over HNSW at equivalent recall on standard ANN benchmarks.

## Gotchas
- The "parallel direction" is query-dependent at inference time but must be approximated at index-build time using representative query statistics or the residual from the nearest centroid.
- λ (the anisotropic weight) is a hyperparameter sensitive to the query distribution; setting λ=1 degenerates to isotropic PQ.
- AISAQ requires knowing the approximate query distribution at build time — out-of-distribution queries at production time can underperform standard PQ.
- The technique applies to MIPS (maximum inner product search), not L2 nearest-neighbor search; for L2, isotropic quantization is already optimal.
- Combining AISAQ with DiskANN (used in production at Microsoft) requires storing anisotropically quantized codes alongside graph adjacency lists — increases index complexity.

## Code pointer
`scann/scann/hashes/internal/asymmetric_hashing2.cc` → `TrainedModel::ScoreNeighbors()`
