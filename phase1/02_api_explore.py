"""Phase 1: Explore Ollama API — list models, check details, compare speeds.

Run: python3 phase1/02_api_explore.py
"""
import os
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import time
import ollama


def list_models():
    """Show all locally available models."""
    models = ollama.list()
    print("=== Installed Models ===")
    for m in models.models:
        size_gb = m.size / (1024**3)
        modified = str(m.modified_at)[:10]
        print(f"  {m.model:30s}  {size_gb:.1f} GB  modified: {modified}")


def benchmark(prompt: str, model: str = "llama3.2:3b"):
    """Measure time-to-first-token (prefill) and generation speed."""
    print(f"\n=== Benchmark: {model} ===")
    start = time.time()
    first_token_time = None
    token_count = 0

    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        if first_token_time is None:
            first_token_time = time.time() - start
        token_count += 1

    total = time.time() - start
    gen_time = total - first_token_time
    print(f"  Time to first token (prefill): {first_token_time:.2f}s")
    print(f"  Tokens generated: {token_count}")
    print(f"  Generation speed: {token_count / gen_time:.1f} tok/s")
    print(f"  Total time: {total:.2f}s")


if __name__ == "__main__":
    list_models()
    benchmark("Write a short poem about machine learning.", "llama3.2:3b")
