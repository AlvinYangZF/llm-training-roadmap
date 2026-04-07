---
id: diskann
year: 2019
tier: advanced
tags: [vector-search, ann, disk, graph, billion-scale, ssd, greedy]
requires: [dense-retrieval, linear-algebra]
summary: "Billion-scale ANN on commodity SSDs using Vamana graph with SSD-resident data."
equations:
  - "robust prune: add edge (u,v) iff ∄w ∈ N_out: α·d(u,w) < d(u,v)"
  - "hops ≈ O(log N), bytes_per_hop = R × d × 4"
complexity: {time: "O(log N) expected hops, each hop = 1 SSD random read", memory: "O(√N) RAM (PQ-compressed), O(N) SSD (full vectors + graph)"}
paper: {title: "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node", authors: "Suhas Jayaram Subramanya et al.", year: 2019}
viz: 17-diskann.html
---

## One-liner
Build a Vamana proximity graph over billion-scale datasets, store most data on SSD, keep only PQ-compressed vectors in RAM, and search with greedy beam traversal issuing one SSD fetch per hop.

## Key equations
```
# Vamana robust prune (builds graph with long + short range edges)
N_out(p) = ∅
for (p*, d*) in sorted(candidates by dist to p):
    if ∃ v ∈ N_out(p): α · d(p, v) ≤ d(p, p*):
        skip                              # p* dominated by closer neighbor v
    N_out(p).add(p*)
    if |N_out(p)| == R: break             # R = max out-degree, typically 64–128

# Two-pass build: pass 1 α=1 (local), pass 2 α>1 (adds long-range highway edges)

# Greedy beam search (beam width L >> k)
frontier = {medoid}
while frontier not exhausted:
    p = closest unvisited node in frontier by PQ distance (RAM)
    fetch N_out(p) full vectors from SSD              # 1 NVMe read ≈ 32–128 KB
    add unvisited neighbors to frontier
return top-k from all visited nodes

# SSD I/O per query
reads_per_query ≈ n_hops × 1   (amortised with prefetch)
bytes_per_hop   = R × d × sizeof(float32)   # 64 × 128 × 4 = 32 KB typical
```

## Why it matters
HNSW requires the entire graph in RAM — a 1-billion-point 128-dim index needs ~512 GB, pricing out commodity servers. DiskANN keeps only 4-byte PQ-compressed vectors (~4 GB for 1B points) in RAM for fast distance estimation; full vectors and graph adjacency lists live on NVMe. The Vamana graph's robust prune creates long-range "highway" edges that drastically cut hop counts (10–30 hops vs. hundreds in flat proximity graphs). Modern NVMe SSDs handle 20–50 random reads per query at sub-millisecond latency each. The result: 1B-scale, 95%+ recall@10, p99 < 5 ms on a single node costing < $10K.

## Gotchas
- SSD endurance matters at high QPS: at 1000 QPS and 20 hops/query, each SSD is hit 72M times/hour — monitor drive health and use enterprise NVMe.
- The graph medoid is computed over the full dataset; in incremental builds the medoid drifts, requiring periodic re-anchoring for stable search quality.
- α > 1 in the second construction pass is critical for good recall on clustered data distributions; default α=1.2, but tuning α ≥ 1.5 increases build time super-linearly.
- PQ distance estimates cause candidate mis-ordering — beam width L must be set 3–5× larger than k to compensate; L=100 for k=10 is a common starting point.
- Logical deletes (tombstoning) are simple; physical compaction to remove deleted nodes requires a partial or full rebuild of affected graph neighbourhoods.

## Code pointer
`DiskANN/src/index.cpp` → `Index<T,TagT,LabelT>::search_with_filters()`
