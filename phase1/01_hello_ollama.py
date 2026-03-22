"""Phase 1, Step 3: Basic Python integration with Ollama.

Run: python3 phase1/01_hello_ollama.py
Prerequisites: ollama pull llama3.2:3b
"""
import os
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import ollama


def chat(prompt: str, model: str = "llama3.2:3b") -> str:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def stream_chat(prompt: str, model: str = "llama3.2:3b"):
    """Stream tokens one by one — watch the KV cache speedup after prefill."""
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print()


if __name__ == "__main__":
    print("=== Basic Chat ===")
    answer = chat("What is a KV cache in transformers? Explain in 3 sentences.")
    print(answer)

    print("\n=== Streaming (watch token-by-token speed) ===")
    stream_chat("Explain quantization for LLMs in 3 sentences.")
