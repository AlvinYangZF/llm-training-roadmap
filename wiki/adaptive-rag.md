---
id: adaptive-rag
year: 2024
tier: applied
tags: [rag, routing, adaptive, query-complexity, classifier]
requires: [rag-pipeline]
summary: "Route queries by complexity: no retrieval, single-step RAG, or iterative RAG."
equations:
  - "c = Classifier(q) ∈ {simple, moderate, complex}"
  - "cost(strategy) = latency + retrieval_calls × k"
complexity: {time: "O(1) classify + O(retrieval_rounds × k) retrieve", memory: "O(model) for classifier, same as base RAG otherwise"}
paper: {title: "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity", authors: "Soyeong Jeong et al.", year: 2024}
viz: 14-adaptive-rag.html
---

## One-liner
A lightweight classifier predicts query complexity and routes each question to the cheapest RAG strategy that can answer it correctly.

## Key equations
```
# Three-way complexity classification
c = argmax_c P(c | q;  θ_clf),   c ∈ {A=no-retrieval, B=single-step, C=iterative}

# Strategy cost (latency proxy)
cost_A = T_gen
cost_B = T_retrieve + T_gen
cost_C = T_retrieve × n_rounds + T_gen × n_rounds

# Classifier training signal (silver labels from oracle outcome)
label(q) = cheapest strategy s* such that answer(q, s*) is correct
         = A if LLM alone is correct
         = B elif single DPR+generate is correct
         = C otherwise

# Expected accuracy under routing policy π
E[acc] = Σ_q P(c=A|q)·acc_A(q) + P(c=B|q)·acc_B(q) + P(c=C|q)·acc_C(q)
```

## Why it matters
Uniform RAG wastes compute on simple factual queries the LLM already knows, while failing on complex multi-hop questions that need iterative retrieval. Adaptive-RAG reduces average latency by routing ~40% of queries to direct generation (no retrieval cost) and reserving expensive iterative retrieval for the 15–20% of queries that genuinely need it. The classifier is small (fine-tuned T5-small or similar) and adds negligible overhead. Training requires only a dataset of questions and their correct answers — silver labels are derived by testing each strategy against a verifier. The result is a Pareto improvement: better accuracy on hard queries, lower latency on easy queries.

## Gotchas
- Silver label quality is the bottleneck: if the verifier (exact match, F1) gives noisy labels, the classifier learns a noisy routing function.
- The three-way split assumes iterative RAG reliably solves multi-hop questions — in practice it can still fail, leaving a residual accuracy gap.
- Distribution shift between training and production query complexity distributions causes the classifier to mis-route; periodic recalibration is needed.
- "No retrieval" for simple queries implicitly assumes the LLM's parametric knowledge is up-to-date; this breaks for time-sensitive facts.
- Iterative RAG (strategy C) has unbounded latency if the stopping criterion is poorly calibrated — always set a hard cap on retrieval rounds.

## Code pointer
`adaptive-rag/src/classifier.py` → `ComplexityClassifier.predict()`
