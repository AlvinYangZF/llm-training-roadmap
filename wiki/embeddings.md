---
id: embeddings
year: 2018
tier: core
tags: [embeddings, word2vec, bert, sentence-transformers, semantic, cosine, dense-vector]
requires: [linear-algebra, tokenization]
summary: "Map tokens or sentences to dense float vectors capturing semantic similarity."
equations:
  - "cos(a,b) = a·b / (‖a‖·‖b‖)"
  - "sim ∈ [-1, 1]"
complexity:
  time: "O(seq_len · d_model) per forward pass"
  memory: "O(V · d) for embedding matrix"
paper:
  title: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
  authors: "Reimers, Gurevych"
  year: 2019
viz: 23-embeddings.html
---

## One-liner
Dense vector representations encode semantic meaning so that similar inputs cluster nearby in ℝ^d under cosine or dot-product distance.

## Key equations
```
cosine similarity:  cos(a, b) = (a · b) / (‖a‖ · ‖b‖)   ∈ [-1, 1]

Word2Vec skip-gram: maximise  Σ_{c ∈ context(w)} log p(c | w)
                    p(c|w) = exp(v_c · v_w) / Σ_j exp(v_j · v_w)

BERT [CLS] pooling: h = Encoder(x_1, ..., x_n)[0]

SBERT mean pooling: h = mean(Encoder(x_1, ..., x_n))   # better than CLS for similarity
```

## Why it matters
Word2Vec (2013) demonstrated that linear structure emerges in embedding space — "king - man + woman ≈ queen" — from a shallow window-prediction objective. BERT made embeddings contextual: the same token "bank" gets different vectors in "river bank" vs "bank account" because the full sequence is processed with self-attention. Sentence-BERT fine-tuned BERT with a siamese network on natural language inference pairs, yielding sentence-level vectors where cosine distance is a reliable similarity proxy. These embeddings are the backbone of semantic search, RAG retrieval, clustering, and classification.

## Gotchas
- BERT [CLS] pooling is poor for sentence similarity out-of-the-box; always use mean pooling or a fine-tuned SBERT model.
- Cosine similarity normalises magnitude — use dot product when magnitude encodes relevance (e.g., ColBERT-style).
- Embedding dimension d (768, 1536, 3072) directly drives ANN index memory and query latency.
- Domain shift: embeddings trained on Wikipedia degrade on medical or legal text; domain-specific fine-tuning is often necessary.
- Matryoshka Representation Learning (MRL) trains embeddings that remain valid when truncated to smaller dimensions, enabling latency/accuracy tradeoffs at query time.

## Code pointer
`sentence_transformers/SentenceTransformer.py` → `SentenceTransformer.encode()`
