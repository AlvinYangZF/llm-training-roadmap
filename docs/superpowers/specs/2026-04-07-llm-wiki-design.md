# LLM Learning Wiki — Design Spec
**Date:** 2026-04-07  
**Status:** Approved  
**Location:** `llm-local/wiki/`

---

## Problem

The project has 29 interactive visualisations, 36+ papers, and a growing learning plan across phases 1–10. Knowledge is scattered: research notes in `research/`, learning plan in `LEARNING_PLAN.md`, no single place for an agent or human to quickly orient, look up a concept, or find prerequisites.

## Goal

A lightweight, single-source wiki that:
1. Any agent (Claude, RAG, scripts) can query programmatically — no LLM call needed for metadata
2. Any human can scan in seconds — Karpathy-minimal prose
3. Supports fast iteration — add a new topic by copying `_TEMPLATE.md` and filling 10 fields
4. Is directly indexable by the existing `rag_agent/agent.py` (FAISS + BM25 on `.md` files)

---

## Structure

```
wiki/
├── INDEX.yaml       ← machine-readable registry of all topics
├── _TEMPLATE.md     ← template for adding new topics
└── <id>.md          ← one file per concept (29 files v1)
```

**No subdirectories.** Flat = zero indirection for agents.

---

## INDEX.yaml Schema

```yaml
version: "1.0"
updated: YYYY-MM-DD
total: N
topics:
  - id: string            # kebab-case, stable identifier
    file: string          # filename relative to wiki/
    year: int             # paper/concept year
    tier: core|advanced|applied
    tags: [string]        # searchable labels
    summary: string       # ≤ 15 words, agent-facing TL;DR
    requires: [string]    # prerequisite topic IDs
    viz: string|null      # linked visualisation page filename
```

### Agent query patterns (no LLM needed)

```bash
# All topics tagged 'memory'
python -c "import yaml; [print(t['id']) for t in yaml.safe_load(open('wiki/INDEX.yaml'))['topics'] if 'memory' in t.get('tags',[])]"

# Prerequisites for flash-attention
python -c "import yaml; t=next(x for x in yaml.safe_load(open('wiki/INDEX.yaml'))['topics'] if x['id']=='flash-attention'); print(t['requires'])"

# Full-text RAG query
python rag_agent/agent.py --docs ./wiki --query "How does PagedAttention handle memory fragmentation?"
```

---

## Per-Topic File Schema (A+C hybrid)

### Frontmatter (machine-readable, mirrors INDEX.yaml)

```yaml
---
id: topic-id
year: YYYY
tier: core|advanced|applied
tags: []
requires: []
summary: "≤15 words."
equations:
  - "key equation 1"
complexity: {time: "O(...)", memory: "O(...)"}
paper: {title: "", authors: "", year: YYYY}
viz: filename.html
---
```

### Prose sections (Karpathy-minimal)

| Section | Purpose | Target length |
|---|---|---|
| `## One-liner` | One sentence, no jargon | 1 sentence |
| `## Key equations` | Fenced code block, no prose | 3–6 lines |
| `## Why it matters` | Impact + context | 3–5 sentences |
| `## Gotchas` | Common mistakes / non-obvious facts | 3–5 bullets |
| `## Code pointer` | Where to find canonical implementation | 1–3 lines |

---

## _TEMPLATE.md

Copy → rename → fill 10 fields. A new topic takes < 5 minutes.

---

## Versioning & Iteration

- Edit any `.md` file directly — no rebuild step
- After adding a topic: add one entry to `INDEX.yaml`, done
- RAG agent auto-reindexes on next run with `--rebuild`
- The `viz` field links back to the visualisation site for interactive exploration

---

## What this is NOT

- Not a blog / tutorial — no long-form explanations
- Not a paper summary repo — `research/` handles that
- Not a replacement for `LEARNING_PLAN.md` — phases live there

---

## Files produced

| File | Purpose |
|---|---|
| `wiki/INDEX.yaml` | Agent entry point, all 29 topics |
| `wiki/_TEMPLATE.md` | Copy-paste template |
| `wiki/*.md` × 29 | One topic file per concept |
