---
id: rag-pipeline
year: 2020
tier: core
tags: [rag, retrieval, generation, pipeline, hyde, dpr, bart]
requires: [dense-retrieval, transformer]
summary: "Retrieval-augmented generation: marginalize over retrieved docs as latent variables."
equations:
  - "p(y|x) = Σ_z p_η(z|x) · p_θ(y|x,z)"
  - "p_η(z|x) = top-k DPR scores, renormalized"
complexity: {time: "O(k · L_gen) generation, O(1) retrieval", memory: "O(N·d) passage index + O(k·S) KV cache for k docs"}
paper: {title: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", authors: "Patrick Lewis et al.", year: 2020}
viz: 13-rag-pipeline.html
---

## One-liner
Augment a seq2seq generator (BART) with a DPR retriever: treat retrieved passages as latent variables and marginalize over them to produce the final answer.

## Key equations
```
# RAG-Token: marginalize over docs at every generation step
p(y_i | x, y_{<i}) = Σ_{z ∈ top-k(x)} p_η(z|x) · p_θ(y_i | x, z, y_{<i})

# RAG-Sequence: pick single doc per full sequence (faster, slightly worse)
p(y | x) = Σ_{z ∈ top-k(x)} p_η(z|x) · p_θ(y | x, z)

# Retrieval distribution (softmax over top-k DPR scores)
p_η(z_i | x) = exp(E_Q(x)^T · E_P(z_i)) / Σ_j exp(E_Q(x)^T · E_P(z_j))

# HyDE variant: generate hypothetical doc, then retrieve doc-to-doc
z_hyp ~ p_θ(z | x),   retrieve by sim(E_P(z_hyp), E_P(p))
```

## Why it matters
Parametric knowledge in LLM weights is static, hallucination-prone, and hard to update. RAG externalises world knowledge to a retrieval index, making facts updatable without retraining and attributable to source documents. Treating retrieved documents as latent variables allows gradient to flow back through both the generator and the retriever, enabling joint fine-tuning. The RAG-Token variant soft-combines evidence from multiple passages, handling questions that require synthesis across sources. HyDE extends the pattern to zero-shot settings where no labelled question-passage pairs exist.

## Gotchas
- RAG-Token marginalisation requires a forward pass per retrieved doc — k=5 means 5× generator compute; batching across docs is essential.
- The retriever and generator can be jointly fine-tuned but the passage index must be periodically refreshed; stale embeddings cause retrieval drift.
- Context window limits constrain k: at 512 tokens/passage and 4096 token context, k≤7 before the generator truncates evidence.
- HyDE generation quality is brittle for highly factual queries — a hallucinated hypothetical doc retrieves irrelevant passages.
- RAG does not automatically resolve conflicting evidence across retrieved passages; the generator may blend contradictory facts without flagging inconsistency.

## Code pointer
`transformers/src/transformers/models/rag/modeling_rag.py` → `RagTokenForGeneration.forward()`
