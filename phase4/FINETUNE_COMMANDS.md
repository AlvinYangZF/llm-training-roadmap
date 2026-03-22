# MLX LoRA Fine-Tuning Command Guide

**Hardware:** Apple M2, 8GB RAM
**Model:** Llama-3.2-3B-Instruct (4-bit quantized)
**Date tested:** 2026-03-22

---

## Step 0: Setup

```bash
cd /Users/zifengyang/Desktop/workspace_claude/llm-local
source .venv/bin/activate

# Install dependencies
pip install mlx-lm huggingface_hub
```

## Step 1: Download Model (~2GB, ~4 minutes)

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'mlx-community/Llama-3.2-3B-Instruct-4bit',
    local_dir='./mlx_models/llama-3.2-3b-4bit'
)
print('Done!')
"
```

## Step 2: Test Base Model (before fine-tuning)

```bash
python3 -m mlx_lm generate \
  --model ./mlx_models/llama-3.2-3b-4bit \
  --prompt "What is a KV cache?" \
  --max-tokens 100
```

**Expected output:** Generic explanation about key-value caches (not transformer-specific)
**Performance:** ~44 tok/s, Peak memory ~1.9 GB

## Step 3: Prepare Training Data

Training data format: JSONL with `messages` field (chat format)
Files must be named `train.jsonl` and `valid.jsonl` in the data directory.

```bash
# If your files have different names, copy them:
cp training_data/sample_train.jsonl training_data/train.jsonl
cp training_data/sample_valid.jsonl training_data/valid.jsonl
```

Example training data entry:
```json
{"messages": [{"role": "user", "content": "What is a KV cache?"}, {"role": "assistant", "content": "A KV cache stores the Key and Value matrices computed during transformer inference..."}]}
```

## Step 4: Fine-tune with LoRA (~50 seconds)

```bash
python3 -m mlx_lm lora \
  --model ./mlx_models/llama-3.2-3b-4bit \
  --train \
  --data ./training_data \
  --batch-size 1 \
  --num-layers 4 \
  --iters 100 \
  --adapter-path ./mlx_models/adapters
```

**Key parameters explained:**
| Parameter | Value | Why |
|-----------|-------|-----|
| `--batch-size 1` | 1 | 8GB RAM constraint, must stay at 1 |
| `--num-layers 4` | 4 | Number of layers to apply LoRA (fewer = less memory) |
| `--iters 100` | 100 | Training iterations (increase for better results) |
| `--adapter-path` | path | Where to save the LoRA adapter weights |

**Actual training metrics from our run:**
```
Trainable parameters: 0.054% (1.737M / 3212.750M)
Training speed: ~2.0 iterations/sec, ~245 tokens/sec
Peak memory: 2.318 GB
Total time: ~50 seconds

Loss progression:
  Iter  10: loss 3.640
  Iter  50: loss 0.979
  Iter 100: loss 0.148  (converged!)

Validation loss: 3.936 → 2.987
```

## Step 5: Test Fine-tuned Model

```bash
# With adapter (fine-tuned)
python3 -m mlx_lm generate \
  --model ./mlx_models/llama-3.2-3b-4bit \
  --adapter-path ./mlx_models/adapters \
  --prompt "What is a KV cache?" \
  --max-tokens 100
```

**Expected output:** Now gives transformer-specific explanation matching training data style

### Before vs After comparison:

**Before (base model):**
> "A KV cache, also known as a Key-Value cache, is a type of caching system that stores data in the form of key-value pairs..."
> (Generic software caching explanation)

**After (fine-tuned):**
> "A KV cache stores the Key and Value matrices computed during transformer inference. Instead of recomputing attention for all previous tokens when generating each new token, the model reuses cached K/V pairs. This makes autoregressive generation much faster — the first token is slow (prefill) but subsequent tokens are fast (decode)."
> (Precise ML-specific explanation matching our training style)

## Step 6: Fuse Adapter into Model (optional)

Merge LoRA weights permanently into the base model:

```bash
python3 -m mlx_lm fuse \
  --model ./mlx_models/llama-3.2-3b-4bit \
  --adapter-path ./mlx_models/adapters \
  --save-path ./mlx_models/llama-3.2-3b-finetuned
```

Output: A standalone 1.7GB model that doesn't need the adapter anymore.

## Step 7: Convert to Ollama (optional)

To use your fine-tuned model with Ollama:

```bash
# Create a Modelfile
cat > Modelfile << 'EOF'
FROM ./mlx_models/llama-3.2-3b-finetuned
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{ .Response }}<|eot_id|>"""
EOF

# Note: Ollama uses GGUF format, so you'd need to convert first.
# For MLX models, it's easier to keep using mlx_lm.generate directly.
```

---

## Tips for Better Results

1. **More training data**: 5 samples is a demo; use 50-500+ for real fine-tuning
2. **More iterations**: Try 200-500 for better convergence
3. **Learning rate**: Default 1e-5 works well; try 5e-6 for stability
4. **Validation monitoring**: Watch val_loss — if it goes up, you're overfitting
5. **Close other apps**: Free RAM for larger batch sizes or more LoRA layers

## Memory Budget (8GB Mac M2)

| Component | Memory |
|-----------|--------|
| macOS + system | ~2-3 GB |
| Model (4-bit 3B) | ~1.9 GB |
| LoRA training overhead | ~0.4 GB |
| **Total during training** | **~2.3 GB peak** |
| **Remaining headroom** | **~3-4 GB** |

---

*All commands verified on Apple M2 8GB, 2026-03-22*
