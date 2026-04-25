---
id: dflash
year: 2026
tier: advanced
tags: [inference, speculative-decoding, diffusion, serving, sglang, vllm]
requires: [decoding, prefill-decode, transformer]
summary: "Block diffusion drafter for speculative decoding: parallel draft blocks, target LLM verifies."
equations:
  - "speedup ~= accepted_tokens / (T_draft_block + T_verify_block)"
  - "accept_prefix = longest prefix where draft token equals target token"
complexity:
  time: "Draft cost is approximately one block forward pass instead of O(k) autoregressive draft steps"
  memory: "Target KV cache plus draft-model KV cache and fused hidden-state context buffers"
paper:
  title: "DFlash: Block Diffusion for Flash Speculative Decoding"
  authors: "Jian Chen, Yesheng Liang, Zhijian Liu"
  year: 2026
viz: 29-dflash.html
---

## One-liner
DFlash accelerates LLM decoding by using a small block diffusion draft model to propose multiple future tokens in parallel, then letting the target LLM verify the draft block so the final output distribution remains unchanged.

## Core idea
Standard autoregressive decoding emits one token at a time. Speculative decoding improves this by asking a smaller draft model to propose several tokens, then verifying those tokens in one target-model pass. The weakness is that many draft models still generate their candidates autoregressively, so the draft stage remains sequential.

DFlash changes the drafter. Instead of producing token 1, then token 2, then token 3, a lightweight block diffusion model fills an entire masked block in one parallel denoising pass. It is not trying to replace the target LLM; it only needs to be accurate enough that many proposed tokens survive target verification.

## Inference flow
```
prompt
  -> target LLM prefill / verification pass
  -> extract hidden states from selected target layers
  -> fuse hidden states through a lightweight projection
  -> inject fused context into every draft-model layer KV cache
  -> block diffusion drafter proposes k tokens in parallel
  -> target LLM verifies the whole block
  -> accept longest matching prefix
  -> repeat until stop condition
```

## Why target conditioning matters
A tiny diffusion drafter by itself lacks the target model's internal reasoning state. DFlash conditions the drafter on hidden features extracted from multiple target layers. Those features are fused, projected, and injected into the draft model's KV cache so every draft layer can attend to target-derived context.

This is the key architectural difference from EAGLE-style drafting: DFlash does not only feed target features into the first draft layer. It keeps the target signal available throughout the draft model, which helps deeper draft models improve acceptance length without losing the latency advantage of parallel block generation.

## Verification preserves quality
DFlash is described as lossless because the target model remains the authority. If the draft block disagrees with the target model, only the matching prefix is accepted and generation continues from the corrected target token. Sampling parameters such as greedy, temperature, top-p, or top-k still belong to the target-model verification path.

## Architecture components
- **Target model**: runs normal prefill and verification, produces hidden states and final accepted tokens.
- **Feature fusion**: selects target hidden states from multiple layers and compresses them into draft conditioning features.
- **KV injection**: writes fused target context into the draft model's per-layer KV cache.
- **Block diffusion drafter**: predicts a fixed-size token block in parallel, often with a small number of draft layers.
- **Verifier**: compares draft tokens against target-model outputs and accepts the longest valid prefix.
- **Serving backend**: SGLang is the reference production path; vLLM has a `DFlashProposer` implementation path; Transformers is useful for experiments.

## Practical tradeoffs
- DFlash helps most when decode latency dominates and generated outputs are long enough to amortize draft overhead.
- Speedup depends on acceptance length. Poor target conditioning, mismatched draft checkpoints, or very high-temperature sampling can reduce gains.
- It requires a trained DFlash draft model matched to the target model family.
- It adds memory for draft weights, draft KV cache, and target hidden-state buffers.
- It complements KV-cache systems such as PagedAttention and disaggregated serving; it does not replace batching, paging, or scheduling.

## Comparison
| Method | Drafter shape | Main bottleneck | Quality guarantee |
|---|---|---|---|
| Vanilla decoding | None | One target pass per token | Native target output |
| Classic speculative decoding | Small autoregressive model | Draft still sequential | Target verifies |
| EAGLE-3 | Feature-conditioned autoregressive drafter | Draft length grows with sequential steps | Target verifies |
| Medusa | Multiple prediction heads | Requires target-side head integration | Target verifies |
| DFlash | Feature-conditioned block diffusion drafter | Verification + acceptance rate | Target verifies |

## Sources
- Paper: [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)
- Project page: [Z Lab DFlash](https://z-lab.ai/projects/dflash/)
- Code: [z-lab/dflash](https://github.com/z-lab/dflash)
- Serving API reference: [vLLM DFlash proposer](https://docs.vllm.ai/en/stable/api/vllm/v1/spec_decode/dflash/)
