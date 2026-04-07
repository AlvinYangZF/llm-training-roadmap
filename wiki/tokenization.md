---
id: tokenization
year: 2016
tier: core
tags: [tokenization, bpe, wordpiece, sentencepiece, vocabulary, subword]
requires: [linear-algebra]
summary: "Convert raw text to integer token IDs via learned subword vocabulary."
equations:
  - "vocab = iterative_merge(byte_pair_counts)"
complexity:
  time: "O(V · n) encode, O(V) table lookup"
  memory: "O(V) vocabulary table"
paper:
  title: "Neural Machine Translation of Rare Words with Subword Units"
  authors: "Sennrich, Haddow, Birch"
  year: 2016
viz: 24-tokenization.html
---

## One-liner
Tokenisers convert raw Unicode text into a fixed-vocabulary sequence of integer IDs that the model's embedding layer can look up.

## Key equations
```
BPE merge:      new_token = most_frequent_adjacent_pair(corpus)
                repeat until |vocab| = V

WordPiece:      merge pair (a, b) if  log P(ab) - log P(a) - log P(b)  is maximised

token count:    n_tokens ≈ n_chars / compression_ratio   (English: ~4 chars/token for BPE-50K)

encode:         text → [id_1, id_2, ..., id_n]   via trie lookup, O(n)
```

## Why it matters
Character-level models are computationally expensive over long sequences; word-level models fail on unknown words and explode vocabulary size. Subword tokenisation — pioneered by BPE for NMT and adopted universally — balances these extremes: common words are single tokens, rare words are decomposed into recognisable subword pieces. This eliminates true OOV (out-of-vocabulary) tokens. SentencePiece decouples tokenisation from whitespace segmentation, making it language-agnostic and suitable for Chinese, Japanese, and other scripts without a tokeniser pre-processing step. Vocabulary size (32K–100K) is a key hyperparameter: larger vocab means shorter sequences but a bigger embedding matrix.

## Gotchas
- BPE is greedy and non-optimal: the same string tokenises differently depending on context boundaries; tiktoken uses byte-level BPE to guarantee no UNK tokens.
- Case sensitivity: GPT tokenisers are case-sensitive; "hello" and "Hello" may be different tokens.
- Tokenisation is not bijective for all schemes — some decodings require offset mapping to recover character positions.
- Numbers and dates tokenise poorly in standard BPE: "2024" might split as ["202", "4"], which can degrade arithmetic reasoning.
- Adding tokens to a pre-trained vocabulary post-hoc (e.g., for a new domain) requires re-initialising those embedding rows and continued pre-training for convergence.

## Code pointer
`tiktoken/core.py` → `Encoding.encode()` / `transformers/tokenization_utils_fast.py` → `PreTrainedTokenizerFast.__call__()`
