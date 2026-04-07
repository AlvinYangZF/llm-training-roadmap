---
id: hnsw
year: 2016
tier: core
tags: [vector-search, ann, graph, navigable-small-world, index]
requires: [dense-retrieval, linear-algebra]
summary: "Hierarchical graph index for approximate nearest neighbour; O(log n) query time."
equations:
  - "layer l: randomly assign nodes, P(node in layer l) = exp(-l/mL)"
  - "query: greedy beam search top-down from coarse to fine layer"
complexity: {time: "O(log n) query, O(n log n) build", memory: "O(n·M) edges, M~16-64"}
paper: {title: "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs", authors: "Malkov & Yashunin", year: 2016}
viz: 19-hnsw.html
---

## One-liner
HNSW builds a multi-layer proximity graph where each node connects to M nearest neighbours; queries navigate from coarse top layers to fine bottom layers in O(log n) steps.

## Key equations
```
# Layer assignment (exponential decay):
l_max(node) = floor(-ln(uniform(0,1)) × mL)   # mL = 1/ln(M), M = max neighbours per node
# Result: ~1/M fraction of nodes in each successive layer

# Insert algorithm:
1. Assign layer l_node
2. Top-down greedy search from entry point to l_node+1: find ef_construction nearest
3. At layers l_node..0: connect node to M closest found, update neighbour connections
4. Heuristic neighbour selection: prefer diverse neighbours (avoid clustering)

# Query algorithm:
W = {entry_point}
for layer L → 0:
    W = search_layer(q, W, ef=1 if layer>0 else ef_search)
    # at each layer: greedy expand W by closest unvisited neighbours
return top-k from W   # ef_search controls recall-speed tradeoff

# Key parameters:
M          = max edges per node (16-64); higher M → better recall, more memory
ef_build   = beam width during index build (100-200); higher → better quality
ef_search  = beam width during query (50-500); tuned at query time for recall target
```

## Why it matters
Flat exhaustive search over million-scale vectors is O(n·d) per query — too slow for real-time RAG. HNSW achieves >95% recall at <1ms latency for million-scale datasets, combining the navigable small world property (short paths between any two nodes) with a hierarchical coarse-to-fine structure. It is the dominant ANN algorithm in production vector databases (Weaviate, Qdrant, pgvector, FAISS). Unlike tree-based indices (KD-tree, ball tree), HNSW degrades gracefully in high dimensions and does not require reindexing on data updates.

## Gotchas
- Memory usage is O(n·M·d·sizeof(id)) for edges plus O(n·d) for vectors — for 100M vectors at M=32 and d=768, edge storage alone is ~25 GB; budget carefully.
- ef_search is a query-time knob: set high for recall-critical workloads (retrieval QA), low for speed-critical workloads (recommendation); benchmark per use case.
- HNSW does not support efficient deletion natively — deleted nodes remain as "tombstones" and degrade recall over time; most DBs do periodic re-indexing.
- The heuristic neighbour selection step is crucial: without it, graph "long-range" shortcuts degrade, and recall collapses at high compression. Don't implement a naive version.
- For cosine similarity, normalise vectors before insertion — inner product and cosine are identical for unit vectors, but HNSW distance function must match the embedding model's training objective.

## Code pointer
`nmslib/hnswlib` → `hnswlib/hnswalg.h` → `HierarchicalNSW<dist_t>::addPoint()` — canonical C++ implementation.
`faiss` → `faiss/IndexHNSW.cpp` → `IndexHNSW::search()` — production-grade version used in Meta's retrieval systems.
