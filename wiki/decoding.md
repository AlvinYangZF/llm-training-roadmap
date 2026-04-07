---
id: decoding
year: 2019
tier: core
tags: [decoding, sampling, beam-search, temperature, top-k, top-p, greedy, repetition]
requires: [gpt2, training-loss]
summary: "Strategies to select tokens from LLM output logits at inference time."
equations:
  - "p_i = exp(z_i/T) / sum exp(z_j/T)"
  - "beam: keep top-k by sum log p(x_t|x_{<t})"
complexity:
  time: "O(V) softmax + O(k log k) top-k sort per step"
  memory: "O(k * seq_len) for beam search state"
paper:
  title: "The Curious Case of Neural Text Degeneration"
  authors: "Holtzman, Buys, Du, Forbes, Choi"
  year: 2020
viz: 28-decoding.html
---

## One-liner
Decoding algorithms translate LLM output logit vectors into discrete token sequences by applying temperature, truncation, and selection strategies at each autoregressive step.

## Key equations
```
greedy:      x_t = argmax_v p(v | x_{<t})

temperature: p_i = exp(z_i / T) / sum_j exp(z_j / T)
             T -> 0: deterministic (greedy)
             T = 1: model distribution
             T > 1: flatter, more random

top-k:       sample from {v : rank(v) <= k}   (k=50 typical)

top-p (nucleus):
             S = smallest set s.t.  sum_{v in S} p(v) >= p
             sample uniformly from S             (p=0.9 typical)

beam search: maintain B hypotheses; at each step expand all, keep top-B by joint log-prob
             score(seq) = (1/|seq|^alpha) * sum_t log p(x_t | x_{<t})   # length penalty

repetition penalty:
             z_i' = z_i / theta   if token i already generated   (theta=1.3 typical)
```

## Why it matters
Greedy decoding is fast but degenerate: it falls into repetitive loops because high-probability tokens self-reinforce. Beam search finds higher-likelihood sequences but produces bland, generic text. Holtzman et al. (2020) showed that human text sits in the high-probability but not maximum-probability region — nucleus sampling (top-p) was designed to stay in that band by dynamically adapting the candidate set size to the probability mass. Temperature scales the sharpness of the distribution without truncation; combining temperature with top-p (and top-k as a hard ceiling) is the standard production configuration.

## Gotchas
- Beam search with B=4-8 is standard for translation/summarisation but not for open-ended generation where diversity matters.
- Temperature T < 0.7 with top-p=1.0 can still produce repetition; pair low T with repetition penalty.
- Top-k is brittle: a fixed k=50 is too broad when the distribution is peaked and too narrow when it is flat; top-p adapts automatically.
- Speculative decoding (draft model proposes, target model verifies in parallel) is orthogonal to these strategies and speeds up sampling without changing output distribution.
- min_p sampling (keep tokens with p_i >= p_min * p_max) is a newer alternative to top-p that avoids sampling near-zero probability tokens while being simpler to implement.

## Code pointer
`transformers/generation/utils.py` → `GenerationMixin.generate()` / `transformers/generation/logits_process.py` → `TopPLogitsWarper.__call__()`
