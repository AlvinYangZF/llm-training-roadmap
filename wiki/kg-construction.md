---
id: kg-construction
year: 2024
tier: applied
tags: [knowledge-graph, extraction, llm, entities, relations, neo4j]
requires: [rag-pipeline, embeddings]
summary: "Extract entity-relation triples from text and store in a graph database."
equations:
  - "triple = (head_entity, relation, tail_entity)"
complexity:
  time: "O(C · L_llm) per chunk, C = chunk count"
  memory: "O(E + R) for graph nodes and edges"
paper:
  title: "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"
  authors: "Edge, Trinh, Cheng, Bradley, Chao, Mody, Truitt, Larson"
  year: 2024
viz: 21-kg-construction.html
---

## One-liner
Prompt an LLM to extract (entity, relation, entity) triples from text chunks, resolve coreferences, merge duplicates, and persist the result as a property graph.

## Key equations
```
triple      = (h, r, t)       h,t ∈ Entities,  r ∈ Relations

entity_id   = canonical(surface_form)           # disambiguation step

graph_score = freq(h, r, t) / total_triples     # edge weight by co-occurrence
```

## Why it matters
Flat vector stores lose structural relationships between entities: knowing that "Turing invented the Turing machine" requires a directed edge, not just proximity in embedding space. Knowledge graphs enable multi-hop reasoning — traversing edges to answer "who influenced researchers who cited Turing?" — which is intractable for pure RAG. GraphRAG (Microsoft, 2024) demonstrated 40–80% better coverage on community-level summarisation questions compared to naive RAG by building a KG as an intermediate index. Entity disambiguation is the hardest step: "Apple" (company) vs "apple" (fruit) requires context-sensitive resolution.

## Gotchas
- LLM extraction recall degrades on long chunks; keep chunks ≤512 tokens for triple extraction.
- Coreference resolution ("He said" → resolve "He" to prior named entity) requires a separate pass or a coreference model.
- Relation vocabulary explosion: unconstrained extraction yields thousands of paraphrase relations; cluster or canonicalise with an ontology.
- Neo4j Cypher queries scan full node labels by default — index entity name properties explicitly.
- Graph construction is one-shot during indexing; real-time updates require incremental triple merging logic.

## Code pointer
`langchain_community/graphs/neo4j_graph.py` → `Neo4jGraph.add_graph_documents()`
