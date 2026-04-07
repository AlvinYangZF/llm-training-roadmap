---
id: graph-rag
year: 2024
tier: applied
tags: [rag, knowledge-graph, community, global-queries, leiden, microsoft]
requires: [rag-pipeline, kg-construction]
summary: "Index corpus as KG, detect communities, answer global queries via map-reduce over summaries."
equations:
  - "G = (V_entities, E_relations)"
  - "answer = Reduce( Map(community_summary_i, query) )"
complexity: {time: "O(E·log E) Leiden detection, O(C) map-reduce at query time", memory: "O(V + E) graph + O(C·S) community summaries"}
paper: {title: "From Local to Global: A Graph RAG Approach to Query-Focused Summarization", authors: "Darren Edge et al.", year: 2024}
viz: 15-graph-rag.html
---

## One-liner
Extract a knowledge graph from the corpus, run Leiden community detection to find thematic clusters, summarize each community, then answer queries by map-reducing over community summaries.

## Key equations
```
# Graph construction (LLM extraction)
V, E = LLM_extract(chunk_i)   for each chunk i
G = ∪_i (V_i, E_i)            merged with entity co-reference resolution

# Leiden community detection (modularity maximization)
Q = (1/2m) Σ_{ij} [A_ij - k_i·k_j / 2m] · δ(c_i, c_j)
C* = argmax_C Q(G, C)

# Hierarchical community summaries
S_c = LLM_summarize({ entity descriptions ∪ relation descriptions } ∈ c)

# Map-reduce query answering
partial_answers = [ LLM_answer(q, S_c) for c in C* ]       # map
final_answer    = LLM_aggregate(partial_answers, q)          # reduce

# Local queries: bypass communities, use vector search over entity/relation text
k-NN(q) over FAISS index of entity + relation embeddings
```

## Why it matters
Standard RAG retrieves locally relevant chunks but cannot answer global questions like "What are the main themes across this entire corpus?" because no single chunk contains that synthesis. Graph-RAG pre-indexes the corpus as a knowledge graph, allowing community summaries to encode corpus-level structure before any query arrives. Leiden community detection is hierarchical, enabling coarse (global) and fine (local) granularity levels. The map-reduce pattern parallelizes well across many community summaries and scales linearly with community count. For local queries (specific entity lookups), Graph-RAG falls back to conventional vector search, making it a strict superset of standard RAG.

## Gotchas
- Graph construction cost is high: extracting entities and relations with an LLM over a large corpus can cost 10–100× more than simple chunking+embedding.
- Entity co-reference resolution is imperfect — "Apple Inc.", "Apple", and "AAPL" may create three nodes unless a merger step is applied.
- Community summaries are static; updating the corpus requires re-running extraction and detection over affected subgraphs, not just re-embedding chunks.
- Leiden is non-deterministic — different runs produce different communities, which can affect answer consistency across deployments.
- Map-reduce over hundreds of community summaries multiplies LLM API cost; set a community relevance threshold before the map step to prune irrelevant communities.

## Code pointer
`graphrag/query/structured_search/global_search/search.py` → `GlobalSearch.asearch()`
