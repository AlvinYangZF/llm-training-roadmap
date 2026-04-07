---
id: lora
year: 2021
tier: applied
tags: [fine-tuning, lora, qlora, low-rank, adapters, peft, rank, merge]
requires: [transformer, training-loss, quantization]
summary: "Fine-tune LLMs by training low-rank weight delta matrices, freezing the base model."
equations:
  - "h = Wx + BAx"
  - "params_lora = 2*d*r  (vs d^2 full)"
  - "ratio = 2r/d  (e.g. r=8, d=4096 = 0.4%)"
complexity:
  time: "O(d*r) adapter forward, same as adding a small linear layer"
  memory: "O(2*d*r) adapter params + O(d^2) frozen weights"
paper:
  title: "LoRA: Low-Rank Adaptation of Large Language Models"
  authors: "Hu, Shen, Wallis, Allen-Zhu, Li, Li, Wang, Chen"
  year: 2021
viz: 27-lora.html
---

## One-liner
Freeze the pretrained weight matrix W and learn a low-rank delta W = BA injected in parallel, updating only 0.1-1% of parameters.

## Key equations
```
forward:          h = Wx + BAx           (W frozen, B and A trained)
                  B in R^{d x r},  A in R^{r x d},  r << d

initialisation:   B = 0,  A ~ N(0, sigma^2)   => delta_W = 0 at step 0

scaling:          h = Wx + (alpha/r) * BAx    (alpha hyper-param, typical alpha = r)

merge at inference: W' = W + (alpha/r) * BA    => zero extra latency

trainable params: 2 * d * r   vs   d^2 full fine-tune
  e.g. d=4096, r=8:  65K  vs  16.8M  (0.4%)
```

## Why it matters
Full fine-tuning a 7B model requires ~56 GB of VRAM for weights plus gradients and optimizer states. LoRA reduces trainable parameters by 100-10000x, cutting optimizer memory proportionally — the Adam states for a rank-8 LoRA adapter on LLaMA-7B are under 100 MB. Crucially, adapters can be merged into the base weights for zero inference overhead, or kept separate for hot-swapping multiple task-specific adapters at runtime. QLoRA (Dettmers et al. 2023) combines LoRA with 4-bit base weights and BF16 adapter training, enabling 65B model fine-tuning on a single 48GB GPU.

## Gotchas
- Rank r is the primary quality/efficiency knob: r=4 is minimal, r=64 approaches full fine-tune quality on complex tasks.
- Apply LoRA to all linear projections (Q, K, V, O, MLP up/gate/down), not just Q and V — restricting to QV misses important adaptation capacity.
- alpha/r scaling: if you double r, double alpha to keep the effective learning rate constant across ranks.
- Merging LoRA into base weights permanently commits the adaptation; keep the unmerged checkpoint for further fine-tuning.
- Multiple LoRA adapters can be composed additively (LoRA-Mix) but interference increases with task dissimilarity.

## Code pointer
`peft/tuners/lora/layer.py` → `Linear.forward()` / `peft/tuners/lora/model.py` → `LoraModel.merge_adapter()`
