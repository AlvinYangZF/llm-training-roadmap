#!/bin/bash
# Phase 4: Fine-tune a small model with MLX LoRA
# This script walks through the MLX fine-tuning workflow on Apple Silicon.
#
# Prerequisites: pip install mlx-lm
# Hardware: Apple M2, 8GB RAM (use small models + small batch size)

set -e

MODEL_DIR="./mlx_models"
DATA_DIR="../training_data"

echo "=== MLX LoRA Fine-Tuning Guide ==="
echo ""
echo "Step 1: Convert a HuggingFace model to MLX format"
echo "  mlx_lm.convert --hf-path meta-llama/Llama-3.2-3B-Instruct -q --upload-repo your-username/model"
echo ""
echo "  Or use a pre-converted model from MLX Community:"
echo "  pip install huggingface_hub"
echo "  huggingface-cli download mlx-community/Llama-3.2-3B-Instruct-4bit --local-dir $MODEL_DIR/llama-3.2-3b-4bit"
echo ""
echo "Step 2: Prepare training data in $DATA_DIR/"
echo "  Format: JSONL with 'messages' field (chat format)"
echo "  See $DATA_DIR/sample_train.jsonl for an example"
echo ""
echo "Step 3: Fine-tune with LoRA"
echo "  mlx_lm.lora \\"
echo "    --model $MODEL_DIR/llama-3.2-3b-4bit \\"
echo "    --train \\"
echo "    --data $DATA_DIR \\"
echo "    --batch-size 1 \\"
echo "    --lora-layers 8 \\"
echo "    --iters 100"
echo ""
echo "Step 4: Test the fine-tuned model"
echo "  mlx_lm.generate \\"
echo "    --model $MODEL_DIR/llama-3.2-3b-4bit \\"
echo "    --adapter-path adapters \\"
echo "    --prompt 'Your test prompt here'"
echo ""
echo "Step 5: Fuse adapter weights into model"
echo "  mlx_lm.fuse \\"
echo "    --model $MODEL_DIR/llama-3.2-3b-4bit \\"
echo "    --adapter-path adapters \\"
echo "    --save-path $MODEL_DIR/llama-3.2-3b-finetuned"
echo ""
echo "=== Notes ==="
echo "- With 8GB RAM, keep batch-size=1 and use 4-bit quantized models"
echo "- 100 iterations is a good starting point; increase for better results"
echo "- Monitor memory with: watch -n1 'memory_pressure'"
