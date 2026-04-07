---
id: dense-retrieval
year: 2020
tier: core
tags: [retrieval, embeddings, dpr, bi-encoder, faiss, contrastive]
requires: [embeddings, linear-algebra]
summary: "Encode questions and passages with separate BERT encoders; retrieve by dot product."
equations:
  - "sim(q, p) = E_Q(q)^T · E_P(p)"
  - "L = -log[ e^{sim(q,p+)} / (e^{sim(q,p+)} + Σ_j e^{sim(q,p_j-)}) ]"
complexity: {time: "O(1) retrieval with FAISS (sub-linear ANN)", memory: "O(N·d) for N passages, d-dim embeddings"}
paper: {title: "Dense Passage Retrieval for Open-Domain Question Answering", authors: "Vladimir Karpukhin et al.", year: 2020}
viz: 12-dense-retrieval.html
---

## One-liner
Represent questions and passages as dense vectors using dual BERT encoders; find the nearest passage to a query using maximum inner product search.

## Key equations
```
# Similarity score (inner product, not cosine — no L2 normalisation in original DPR)
sim(q, p) = E_Q(q)^T · E_P(p)

# In-batch negatives contrastive loss (NLL over batch of B pairs)
L = -1/B · Σ_i log [ exp(sim(q_i, p_i+)) /
                     (exp(sim(q_i, p_i+)) + Σ_{j≠i} exp(sim(q_i, p_j))) ]

# FAISS MIPS index (exact)
k-NN(q) = argtop-k_p { E_Q(q)^T · E_P(p) }

# FAISS IVF-PQ (approximate, production)
p̂ = argtop-k over centroid-filtered PQ-compressed embeddings
```

## Why it matters
Sparse retrieval (BM25, TF-IDF) fails on lexical mismatch — a question about "heart attack" doesn't match a passage about "myocardial infarction". DPR trains encoders end-to-end with question-passage pairs so semantic similarity is captured in geometry. In-batch negatives are the key efficiency trick: a batch of B positive pairs yields B²-B free negatives, making training tractable without a large negative mining pipeline. Once passage embeddings are precomputed and indexed with FAISS, retrieval latency is sub-millisecond regardless of corpus size. DPR established the bi-encoder paradigm that underlies virtually every modern RAG retriever.

## Gotchas
- DPR uses inner product, not cosine similarity — L2-normalising embeddings changes the ranking and is not equivalent unless you renormalize both encoders.
- In-batch negatives assume random shuffling; if a batch accidentally contains a hard negative that is actually a positive, the loss is noisy — use gold-verified negatives or BM25 hard negatives.
- The passage encoder and question encoder are initialised from the same BERT checkpoint but fine-tuned independently; sharing weights hurts retrieval quality.
- FAISS IVF requires a training phase on corpus embeddings before indexing; skipping this produces poor cluster assignments and degrades recall.
- DPR retrieval recall@k saturates around k=100; pushing k higher gives diminishing returns and increases re-ranker or reader latency.

## Code pointer
`DPR/dpr/models/biencoder.py` → `BiEncoder.forward()`
