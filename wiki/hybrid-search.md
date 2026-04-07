---
id: hybrid-search
year: 2024
tier: applied
tags: [retrieval, bm25, dense, rrf, fusion, sparse, lexical]
requires: [dense-retrieval, hnsw]
summary: "Combine BM25 sparse retrieval with dense vector search via RRF fusion."
equations:
  - "RRF(d) = Σ_i 1 / (k + rank_i(d))"
complexity:
  time: "O(n log n) per ranker + O(R) fusion"
  memory: "O(n) inverted index + O(n·d) vector store"
paper:
  title: "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
  authors: "Cormack, Clarke, Buettcher"
  year: 2009
viz: 20-hybrid-search.html
---

## One-liner
Merge BM25 lexical rankings and dense vector rankings using Reciprocal Rank Fusion to get the best of both retrieval paradigms.

## Key equations
```
RRF score:   score(d) = Σ_i  1 / (k + rank_i(d))

BM25 score:  BM25(q, d) = Σ_{t∈q} IDF(t) · (tf(t,d) · (k1+1)) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl))

k=60  (RRF constant, empirically robust)
```

## Why it matters
BM25 excels at exact keyword match — critical for rare terms, product codes, and named entities where semantic proximity is meaningless. Dense retrieval captures semantic similarity and handles paraphrase, synonym, and multilingual queries. Neither alone dominates across all query types. RRF fusion requires no score normalisation: rank positions are ordinal, not cardinal, so scores from heterogeneous systems combine safely. The k=60 constant dampens the influence of very high rank differences and was found optimal across TREC benchmarks.

## Gotchas
- RRF assumes both rankers return the same candidate pool size; truncating one list early degrades fusion quality.
- BM25 is sensitive to tokenisation: stemming and stopword lists must match between indexing and query time.
- Dense retrieval recall is bounded by HNSW approximate search; ANN misses propagate into final results.
- Tuning RRF weights (weighted RRF: `α/( k+rank_dense) + β/(k+rank_bm25)`) requires held-out queries; default equal weighting is a safe baseline.
- Hybrid pipelines double latency if rankers run sequentially; parallelise retrieval before fusion.

## Code pointer
`langchain/retrievers/ensemble.py` → `EnsembleRetriever.get_relevant_documents()`
