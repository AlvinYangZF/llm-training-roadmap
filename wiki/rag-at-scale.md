---
id: rag-at-scale
year: 2022
tier: applied
tags: [rag, production, chunking, reranking, hybrid, latency, eval]
requires: [rag-pipeline, hybrid-search]
summary: "Production RAG: chunk sizing, overlap, reranking, hybrid search, and RAGAS evaluation."
equations:
  - "hybrid_score = α·BM25(q,d) + (1-α)·dense_sim(q,d)"
  - "RAGAS = f(faithfulness, answer_relevance, context_recall)"
complexity: {time: "O(R·log N) BM25 + O(1) FAISS + O(R·L²) cross-encoder rerank", memory: "O(N·d) dense index + O(N·V) BM25 inverted index"}
paper: {title: "RAGAS: Automated Evaluation of Retrieval Augmented Generation", authors: "Shahul Es et al.", year: 2023}
viz: 16-rag-at-scale.html
---

## One-liner
Production RAG requires deliberate decisions on chunk size, overlap, retrieval strategy, reranking, context ordering, and continuous evaluation — each choice has measurable latency and quality tradeoffs.

## Key equations
```
# Hybrid retrieval (reciprocal rank fusion or linear blend)
RRF_score(d, Q) = Σ_{r ∈ {BM25,dense}} 1 / (k + rank_r(d))
hybrid_score    = α · BM25_norm(q,d) + (1-α) · cosine(E_Q(q), E_P(d))

# Cross-encoder reranking (retrieve top-R, rerank to top-K)
score(q, d) = CrossEncoder([CLS] q [SEP] d [SEP])
top_K = argtop-K_{d ∈ top-R} score(q, d)      # R=20, K=5 typical

# RAGAS evaluation metrics
faithfulness      = |supported claims| / |total claims in answer|
answer_relevance  = avg cosine(E(generated_q_i), E(original_q))   # reverse generation
context_recall    = |relevant sentences in context| / |relevant sentences in ground truth|
RAGAS = harmonic_mean(faithfulness, answer_relevance, context_recall)

# Lost-in-middle mitigation: reorder retrieved docs
order = [top_1, top_3, top_5, ..., top_4, top_2]   # most relevant at edges
```

## Why it matters
RAG quality is highly sensitive to engineering choices that papers rarely report. Chunk size 512 tokens with 64-token overlap balances context coherence against retrieval precision — smaller chunks improve precision but lose cross-sentence context; larger chunks reduce precision and hit context window limits. Hybrid BM25+dense retrieval consistently outperforms either alone, especially for queries with rare proper nouns or exact string matches that dense models mishandle. Cross-encoder reranking at retrieve-20/rerank-5 recovers 5–15% NDCG over pure bi-encoder retrieval at manageable latency cost. RAGAS provides a reference-free eval loop, enabling continuous quality monitoring without human annotation.

## Gotchas
- Chunk boundaries should respect natural document structure (paragraphs, sections), not fixed token counts — splitting mid-sentence degrades coherence.
- The α blend weight in hybrid retrieval must be tuned per domain; default 0.5 is rarely optimal, particularly for code or medical corpora where BM25 has high recall.
- Cross-encoder reranking latency scales with R×avg_passage_length; measure wall-clock p99 before committing to R=20 in production.
- RAGAS faithfulness requires an LLM-as-judge call per evaluation sample — at scale, this becomes expensive; subsample or cache evaluations.
- "Lost in the middle" is a real phenomenon: LLMs better utilise context at the start and end of their window — always place highest-ranked passages at position 0 and -1.

## Code pointer
`langchain/retrievers/ensemble.py` → `EnsembleRetriever.get_relevant_documents()`
