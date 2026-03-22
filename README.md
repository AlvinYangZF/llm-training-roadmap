# LLM Training from Scratch — Learning Roadmap

A hands-on, progressive learning path from running local LLMs to training language models from scratch. Built on Apple Silicon (Mac mini M2 8GB), scaling to Mac Studio and AWS cloud GPUs.

## Current Progress

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1: First Local LLM | ✅ Done | Ollama chat, API exploration, streaming |
| Phase 2: Core Concepts | ✅ Done | Transformer, tokenization, attention, KV cache, quantization |
| Phase 3: RAG & Chains | ✅ Done | LangChain chains, RAG pipeline → upgraded to v2 (hybrid search + reranking) |
| Phase 4: Fine-tuning Intro | ✅ Done | MLX LoRA fine-tuning Llama 3.2 3B 4-bit (loss 3.640 → 0.148) |
| Phase 5: Advanced Fine-tuning | ⬜ Next | Data preparation, QLoRA sweeps, evaluation metrics |
| Phase 6: Training Theory | ⬜ | Pre-training data pipeline, tokenizer training, scaling laws |
| Phase 7: Train from Scratch | ⬜ | nanoGPT, MLX training, 50M param model on Mac |
| Phase 8: Mac Studio Scale-up | ⬜ | Hardware upgrade, 1B model training |
| Phase 9: Cloud Training (AWS) | ⬜ | Multi-GPU distributed training, 1-7B pre-training |
| Phase 10: Production & Advanced | ⬜ | RLHF/DPO, distillation, merging, HuggingFace publishing |

## Quick Start

```bash
# 1. Set up environment
cd llm-local
python3 -m venv .venv
source .venv/bin/activate
pip install langchain langchain-ollama langchain-community langchain-text-splitters \
    faiss-cpu flashrank rank_bm25 mlx mlx-lm transformers pypdf docx2txt

# 2. Verify Ollama is running
ollama list

# 3. Run Phase 1 — chat with local model
python3 phase1/01_hello_ollama.py

# 4. Run RAG agent (Phase 3, upgraded to v2)
python3 rag_agent/agent.py --docs ./path/to/docs --query "Your question here"
```

## Project Structure

```
llm-local/
├── phase1/                      # First Local LLM
│   ├── 01_hello_ollama.py       # Basic chat + streaming
│   └── 02_api_explore.py        # List models, benchmark speed
├── phase2/                      # Core Concepts
│   └── concepts.py              # Interactive concept explorer (8 topics)
├── phase3/                      # RAG & Chains
│   ├── 01_langchain_chain.py    # Prompt templates + chains
│   └── 02_rag_pipeline.py       # RAG with FAISS + local embeddings
├── phase4/                      # Fine-tuning
│   ├── 01_mlx_finetune.sh       # MLX LoRA fine-tuning script
│   └── FINETUNE_COMMANDS.md     # Step-by-step fine-tuning guide
├── rag_agent/                   # Production RAG Agent v2
│   └── agent.py                 # Hybrid BM25+FAISS, FlashRank reranking
├── training_data/               # Sample fine-tuning data
│   ├── train.jsonl
│   ├── valid.jsonl
│   ├── sample_train.jsonl
│   └── sample_valid.jsonl
├── LEARNING_PLAN.md             # Full 54-exercise learning plan (Phases 5-10)
└── outputs/                     # Generated outputs
```

## Models Used

| Model | Size | Purpose |
|-------|------|---------|
| `llama3.2:3b` | ~2GB | General chat, fine-tuning base |
| `deepseek-r1:8b` | ~5GB | Reasoning tasks |
| `bge-m3` | ~1.2GB | Embeddings (upgraded from nomic-embed-text) |
| `nomic-embed-text` | ~275MB | Embeddings (fallback) |

## RAG Agent v2 Features

The RAG agent (`rag_agent/agent.py`) was optimized with P0 improvements:

- **Chunking**: 512 tokens, 64 overlap (benchmark-validated sweet spot)
- **Embedding**: BGE-M3 (higher MTEB score, better multilingual)
- **Retrieval**: Hybrid BM25 (sparse) + FAISS (dense) with Reciprocal Rank Fusion
- **Reranking**: FlashRank neural reranker (retrieve top-20 → rerank to top-5)
- **Context**: Reordering to mitigate "lost in the middle" problem

```bash
# Basic query
python3 rag_agent/agent.py --docs ./docs --query "What is PagedAttention?"

# Interactive chat
python3 rag_agent/agent.py --docs ./docs

# Options
python3 rag_agent/agent.py --docs ./docs --model deepseek-r1:8b  # different LLM
python3 rag_agent/agent.py --docs ./docs --embed-model nomic-embed-text  # different embeddings
python3 rag_agent/agent.py --docs ./docs --no-rerank  # faster, skip reranking
python3 rag_agent/agent.py --docs ./docs --rebuild  # force re-index
```

## Learning Plan Overview

**Total: 54 exercises, ~5-7 months, $4,000-8,000**

### Phase 5: Advanced Fine-tuning (3-4 weeks, $0, Mac mini)
Data preparation mastery, LoRA rank/alpha sweeps, QLoRA mechanics, perplexity evaluation, multi-task fine-tuning, catastrophic forgetting experiments.

### Phase 6: Training from Scratch — Theory (3-4 weeks, $0, Mac mini)
Pre-training data pipeline (RefinedWeb), BPE tokenizer training (SentencePiece), loss functions, cosine LR schedule, gradient accumulation, mixed precision, distributed training concepts (DDP/FSDP/ZeRO), Chinchilla scaling laws.

### Phase 7: Train a Small LLM from Scratch (4-5 weeks, $0, Mac mini)
nanoGPT study and training, hyperparameter experiments, MLX-native training, character-level LM, full pipeline capstone (50M param model, 8-24hr training run).

### Phase 8: Mac Studio Upgrade (1-2 weeks, ~$3,500-4,500)
Hardware selection (M4 Max 128GB recommended), MLX vs PyTorch benchmarking, 1B model training (3-7 day runs).

### Phase 9: Cloud Training — AWS (4-6 weeks, $500-2,000)
GPU instance selection, spot instance checkpointing, HuggingFace Trainer, DeepSpeed ZeRO, FSDP, capstone: pre-train 1B model from scratch on 8×A100.

### Phase 10: Production & Advanced (6-8 weeks, $500-1,000)
SFT + DPO alignment, lm-eval-harness benchmarking, model merging (SLERP/TIES/DARE), knowledge distillation, continuous pre-training, synthetic data generation, HuggingFace publishing.

See [LEARNING_PLAN.md](LEARNING_PLAN.md) for the full detailed plan with all 54 exercises.

## Hardware Path

| Stage | Hardware | What's Possible |
|-------|----------|-----------------|
| Current | Mac mini M2 8GB | LoRA 3B, train ≤150M from scratch |
| Next | Mac Studio M4 Max 128GB | LoRA 13B, train 1-3B from scratch |
| Scale | AWS p4d (8×A100) | Pre-train 1-7B, distributed training |

## Key Papers Studied

36 papers on LLM inference optimization, organized by topic:
- **Paging & KV Management**: PagedAttention/vLLM
- **Disaggregated Serving**: Mooncake (FAST'25 Best Paper), DistServe
- **Attention Optimization**: FlashAttention 1&2, FlashInfer, MQA, GQA
- **KV Quantization**: KIVI, KVQuant, CoupledQuantization
- **Intelligent Eviction**: H2O, Scissorhands, StreamingLLM, SnapKV
- **Offloading**: FlexGen, InfiniGen, KVSwap
- **Caching**: SGLang/RadixAttention, LMCache, CachedAttention
- **Distributed**: InfiniteLLM, DejaVu, TraCT

## License

MIT
