# LLM Training Learning Plan — Phases 5-10

## Context

You've completed Phases 1-4: Ollama serving, transformer concepts, RAG pipeline, and basic LoRA fine-tuning (Llama 3.2 3B 4-bit, loss 3.640→0.148 on Mac mini M2 8GB). You have 36 papers studied and a working RAG system. This plan takes you from fine-tuning practitioner to training LLMs from scratch.

**Hardware path:** Mac mini M2 8GB → Mac Studio (128-192GB) → AWS cloud GPUs

---

## Phase 5: Advanced Fine-tuning on Mac Mini (3-4 weeks, $0)

### 5.1 Data Preparation Mastery
- **Exercise 1:** Convert a raw dataset (OpenAssistant oasst1) into ChatML, Alpaca, and ShareGPT formats
- **Exercise 2:** Build a data quality pipeline: MinHash dedup (`datasketch`), length filtering, perplexity scoring, language detection (`langdetect`). Target: reduce noisy 10K+ dataset by 20-40%
- **Exercise 3:** Dataset mixing — combine instruction, domain-specific, and conversation data with configurable ratios
- **Key paper:** "LIMA: Less Is More for Alignment" — 1000 high-quality examples can outperform 50K noisy ones

### 5.2 QLoRA and Advanced LoRA Configurations
- **Exercise 4:** LoRA rank/alpha sweep — rank {4,8,16,32,64} with matched alpha. Record loss, time, adapter size, generation quality. Find the sweet spot (~rank 8-16 for 3B)
- **Exercise 5:** Target module experiments — (a) q+v only, (b) all attention, (c) attention+MLP. Compare quality vs memory tradeoffs
- **Exercise 6:** Understand QLoRA mechanics — 4-bit NF4 base + bfloat16 adapters. Verify memory usage differences

**8GB M2 limits:**
| Model | Max LoRA Rank | Batch Size | Max Seq Len |
|-------|--------------|------------|-------------|
| Llama 3.2 3B 4-bit | 32 | 1 | 512-1024 |
| Llama 3.2 1B 4-bit | 64 | 1-2 | 2048 |
| 7B+ | Not feasible | — | — |

### 5.3 Evaluation Metrics
- **Exercise 7:** Compute perplexity on held-out test set — compare base vs fine-tuned (expect 20-60% improvement in-domain)
- **Exercise 8:** Task-specific metrics — ROUGE for summarization, pass@k for code, exact match/F1 for QA
- **Exercise 9:** Blind human evaluation — generate outputs from base and fine-tuned model on 20 prompts, rate 1-5

### 5.4 Multi-task Fine-tuning
- **Exercise 10:** Sequential fine-tuning (A then B) — observe catastrophic forgetting
- **Exercise 11:** Mixed fine-tuning (A+B shuffled) — compare to sequential approach

**Checkpoint:** Can prepare data, tune LoRA hyperparameters, evaluate with perplexity + task metrics, explain QLoRA mechanics

---

## Phase 6: Training from Scratch — Theory (3-4 weeks, $0)

### 6.1 Pre-training Data Pipeline
- **Exercise 12:** Study RefinedWeb paper — write 2-page summary of: URL filtering → text extraction → language ID → quality filtering (KenLM) → MinHash dedup → PII removal
- **Exercise 13:** Train a BPE tokenizer from scratch on 100MB Wikipedia text using SentencePiece. Compare vocab sizes 8K/16K/32K — measure tokens-per-word efficiency
- **Exercise 14:** Compare your tokenizer vs LLaMA's on 1000 sentences — analyze code/URL/multilingual handling

### 6.2 Training Objectives & Loss
- **Exercise 15:** Write one-page explanation of causal LM vs masked LM, causal attention mask, teacher forcing
- **Exercise 16:** Loss function deep dive — understand what loss values mean: loss 3.0 ≈ 5% probability on correct token, loss 2.0 ≈ 13%, loss 1.5 ≈ 22%

### 6.3 Training Dynamics
- **Exercise 17:** Implement cosine decay with warmup LR schedule — plot it with matplotlib
- **Exercise 18:** Write a gradient accumulation training loop from scratch in PyTorch (~20 lines). `effective_batch = micro_batch × accum_steps × num_GPUs`
- **Exercise 19:** Study mixed precision — write summary of float16 vs bfloat16 (bfloat16 has same exponent range as float32, no loss scaling needed)

### 6.4 Distributed Training Concepts (theory only)
- **Exercise 20:** DDP — each GPU has full model, all-reduce gradients
- **Exercise 21:** FSDP/ZeRO — shard parameters, gradients, optimizer states across GPUs
- **Exercise 22:** Tensor parallelism (split layers) vs Pipeline parallelism (split model vertically)

### 6.5 Scaling Laws
- **Exercise 23:** Chinchilla scaling laws — tokens ≈ 20× parameters. C ≈ 6ND. Work through math for your planned training runs

**Key papers:** ZeRO, Megatron-LM, Chinchilla, Kaplan Scaling Laws
**Key reading:** Lilian Weng's "Large Language Model Training Techniques"

**Checkpoint:** Can train a tokenizer, write LR schedule + gradient accumulation from scratch, explain DDP vs FSDP, compute optimal model size for given compute budget

---

## Phase 7: Train a Small LLM from Scratch on Mac (4-5 weeks, $0)

### 7.1 nanoGPT
- **Exercise 24:** Clone `karpathy/nanoGPT` — read and annotate every line of `model.py` and `train.py`
- **Exercise 25:** Train on Shakespeare (~1MB): 6 layers, 6 heads, 384 dim, ctx 256 → ~10M params. Train 30-60 min on M2. Expected loss: ~1.0-1.2 (char-level)
- **Exercise 26:** Hyperparameter experiments — vary one at a time: LR {1e-3, 3e-4, 1e-4, 3e-5}, model size {3M, 10M, 30M}, context {64, 128, 256, 512}, batch size {8, 32, 128}. Document all results in a table
- **Exercise 27:** Train on a different dataset (code, Wikipedia, custom corpus) — compare loss curves and generation quality

**Feasible model sizes on 8GB M2:**
| Params | Precision | Batch Size | Status |
|--------|-----------|------------|--------|
| 50M | float32 | 4 | Comfortable |
| 100-120M | float32 | 1 + grad checkpoint | Tight but feasible |
| 150M | bfloat16 | 1 + grad checkpoint | Max possible |
| 300M+ | Any | — | Not feasible |

### 7.2 MLX-native Training
- **Exercise 28:** Port nanoGPT to MLX — understand lazy evaluation vs PyTorch eager mode
- **Exercise 29:** Benchmark PyTorch vs MLX on same 10M model — expect MLX 1.5-3× faster on Apple Silicon
- **Exercise 30:** Train a character-level LM (no tokenizer, vocab ~100-200) — watch it learn spelling → words → grammar → style

### 7.3 Full Pipeline Capstone
- **Exercise 31:** End-to-end mini pre-training:
  1. Collect 50-100MB corpus
  2. Train SentencePiece tokenizer (vocab 8192)
  3. Implement data loading + train/val split
  4. Define 50M param GPT (8 layers, 8 heads, 512 dim, ctx 512)
  5. Train with: AdamW, cosine LR + warmup, gradient accumulation, bfloat16
  6. Log loss, val loss, LR, tokens/sec every 500 steps
  7. Generate samples at checkpoints
  8. **Expected time: 8-24 hours on Mac mini M2**

**Key resources:** `karpathy/nanoGPT`, `ml-explore/mlx-examples`, Karpathy's "Let's build GPT" YouTube video

**Checkpoint:** Trained a GPT from random init to coherent text, can explain every component, tuned hyperparameters from training dynamics

---

## Phase 8: Mac Studio Upgrade Path (1-2 weeks research + ongoing)

### Hardware Recommendations
| Config | Memory | Price (USD) | Best For |
|--------|--------|-------------|----------|
| Mac Studio M2 Ultra 192GB | 192GB | $4,000-5,000 (refurb) | Maximum memory ceiling |
| **Mac Studio M4 Max 128GB** | 128GB | $3,500-4,500 (new) | **Recommended sweet spot** |
| Mac Studio M4 Max 64GB | 64GB | $2,500-3,000 (new) | Budget option |

### What Becomes Possible
| Task | 64GB | 128GB | 192GB |
|------|------|-------|-------|
| Fine-tune 7B QLoRA | Comfortable | Comfortable | Comfortable |
| Fine-tune 13B QLoRA | Tight | Comfortable | Comfortable |
| Pre-train 1B from scratch | Comfortable | Comfortable | Comfortable |
| Pre-train 3-7B (bf16) | Not feasible | Feasible + grad ckpt | Comfortable |
| Fine-tune 70B QLoRA | Not feasible | Not feasible | Feasible but slow |

### Exercises
- **Exercise 32:** Benchmark MLX vs PyTorch at 100M, 500M, 1B model sizes
- **Exercise 33:** Train 1B model from scratch on 5-20B tokens (3-7 day run)
- **Exercise 34:** Repeat Phase 7 capstone at 500M-1B scale
- **Exercise 35:** Full-parameter fine-tuning (no LoRA) on 3B model — compare to LoRA results

**Cost comparison:** Mac Studio pays for itself after ~125 hours of equivalent cloud GPU time, but cloud A100s are 5-10× faster per dollar-hour for large runs

---

## Phase 9: Cloud Training — AWS (4-6 weeks, $500-2,000)

### Instance Selection
| Instance | GPU | GPU Mem | Spot $/hr | Best For |
|----------|-----|---------|-----------|----------|
| g5.xlarge | 1× A10G | 24GB | $0.30-0.50 | Fine-tuning 7B QLoRA |
| g5.12xlarge | 4× A10G | 96GB | $1.70-2.50 | Pre-training 1-3B |
| p4d.24xlarge | 8× A100 | 320GB | $9.80-15.00 | Pre-training 3-7B |
| p5.48xlarge | 8× H100 | 640GB | $30-50 | Pre-training 7B+ |

### Cost Estimates
| Run | Instance | Duration | Spot Cost |
|-----|----------|----------|-----------|
| Fine-tune 7B QLoRA | 1× A10G | 4-10 hrs | $5-15 |
| Pre-train 1B / 20B tokens | 4× A10G | 4-8 days | $200-400 |
| Pre-train 3B / 60B tokens | 8× A100 | 5-10 days | $2,000-4,000 |
| Pre-train 7B / 140B tokens | 8× A100 | 14-21 days | $8,000-15,000 |

### Exercises
- **Exercise 36:** Set up AWS Deep Learning AMI, install stack (transformers, accelerate, deepspeed, wandb)
- **Exercise 37:** Implement robust checkpointing for spot instances — save model, optimizer, scheduler, data position, RNG states every 1000 steps. Test kill/resume
- **Exercise 38:** HuggingFace Trainer — randomly initialized model, full pipeline, W&B logging
- **Exercise 39:** Accelerate + DeepSpeed ZeRO Stage 2 on 4-8 GPUs — verify ~3.5-3.8× scaling
- **Exercise 40:** PyTorch FSDP via Accelerate — compare to DeepSpeed ZeRO-3

### Capstone: Pre-train 1B from Scratch
- **Exercise 41:** Full 1B pre-training run:
  - Architecture: LLaMA-style (RMSNorm, SwiGLU, RoPE, GQA) — 16 layers, 2048 dim, 8 heads
  - Data: 20-50B tokens from FineWeb-Edu
  - Config: 8× A100 spot, ZeRO Stage 2, ~2M tokens/batch, LR 3e-4 cosine, bf16
  - Cost: $500-1,500 over 3-7 days
- **Exercise 42:** Evaluate with lm-eval-harness (HellaSwag, ARC, MMLU) — compare to TinyLlama/Pythia 1B

**Checkpoint:** Can set up multi-GPU cloud training, handle spot interruptions, train 1B+ model, benchmark against published results

---

## Phase 10: Production & Advanced Topics (6-8 weeks, $500-1,000)

### 10.1 Alignment (RLHF/DPO)
- **Exercise 43:** SFT stage — fine-tune pre-trained 1B on instruction data (OpenAssistant/UltraChat)
- **Exercise 44:** DPO alignment — train with `trl.DPOTrainer` on preference pairs. Compare SFT vs SFT+DPO
- **Exercise 45:** Study RLHF pipeline (reward model → PPO). Understand DPO is preferred in practice

### 10.2 Evaluation
- **Exercise 46:** Run lm-eval-harness: HellaSwag, ARC, MMLU, TruthfulQA, Winogrande, GSM8K
- **Exercise 47:** HumanEval for code models — measure pass@1 and pass@10
- **Exercise 48:** Build custom domain-specific evaluation suite (100+ questions)

### 10.3 Model Merging
- **Exercise 49:** Study merging: LERP, SLERP, TIES, DARE
- **Exercise 50:** Use `mergekit` to merge two fine-tuned models (code + instruction). Evaluate combined capabilities

### 10.4 Knowledge Distillation
- **Exercise 51:** Distill 7B teacher → 1B student with mixed cross-entropy + KL divergence loss. Expect 5-15% benchmark improvement over training from scratch

### 10.5 Continuous Pre-training
- **Exercise 52:** Continue pre-training Llama 3.2 1B on 1-5B domain tokens. Use 70% domain + 30% general data to prevent forgetting. Use 1/10th original peak LR

### 10.6 Synthetic Data
- **Exercise 53:** Generate 10K instruction-response pairs with a strong model. Filter, fine-tune your 1B, compare to human-written data

### 10.7 Publishing
- **Exercise 54:** Write model card, upload to HuggingFace Hub with weights, tokenizer, config, eval results

---

## Timeline Summary

| Phase | Duration | Hardware | Cost |
|-------|----------|----------|------|
| 5: Advanced Fine-tuning | 3-4 weeks | Mac mini M2 8GB | $0 |
| 6: Training Theory | 3-4 weeks | Mac mini M2 8GB | $0 |
| 7: Train from Scratch (Small) | 4-5 weeks | Mac mini M2 8GB | $0 |
| 8: Mac Studio Upgrade | 1-2 weeks + ongoing | Mac Studio 128-192GB | $3,000-5,000 |
| 9: Cloud Training | 4-6 weeks | AWS p4d/g5 spot | $500-2,000 |
| 10: Production Topics | 6-8 weeks | Mac Studio + AWS | $500-1,000 |
| **Total** | **~5-7 months** | | **$4,000-8,000** |

**Phases 5-7 are free and use your current Mac mini.** That's 10-13 weeks of learning before any spend. By the end of Phase 7, you will have trained a language model from scratch — the fundamental milestone.

## Verification

After each phase, verify understanding by:
1. Completing all exercises and documenting results in dated files under `research/`
2. Meeting the checkpoint criteria listed at the end of each phase
3. Building on the previous phase's code/models (each phase uses outputs from the last)
