"""Phase 2: Interactive concept explorer — ask the local LLM to teach you.

Run: python3 phase2/concepts.py
"""
import os
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import ollama

CONCEPTS = {
    "1": ("Transformer Architecture", "Explain the transformer architecture (encoder/decoder, self-attention) as if I'm a developer who knows Python but not ML."),
    "2": ("Tokenization", "Explain tokenization (BPE, SentencePiece) in LLMs. Show a concrete example of how 'Hello world' might be tokenized."),
    "3": ("Attention Mechanism", "Explain the Q/K/V attention mechanism step by step. Use a simple analogy."),
    "4": ("KV Cache", "Explain KV cache in transformer inference. Why does the first token take longer than subsequent tokens? What gets cached and why?"),
    "5": ("Quantization", "Explain quantization (FP32 → FP16 → INT8 → INT4). How does it reduce model size? What's the quality tradeoff?"),
    "6": ("Context Window", "Explain context windows in LLMs. What determines the max context length? What happens if input exceeds it?"),
    "7": ("Temperature & Top-p", "Explain temperature and top-p sampling. Give examples of when you'd use high vs low temperature."),
    "8": ("GGUF Format", "Explain the GGUF model format used by llama.cpp and Ollama. How is it different from safetensors or PyTorch .bin files?"),
}


def explore(topic_key: str, model: str = "llama3.2:3b"):
    name, prompt = CONCEPTS[topic_key]
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}\n")

    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful ML tutor. Be concise but thorough. Use analogies and examples."},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print("\n")


if __name__ == "__main__":
    print("=== LLM Concept Explorer ===")
    print("Pick a topic to learn about:\n")
    for k, (name, _) in CONCEPTS.items():
        print(f"  {k}. {name}")
    print(f"  a. All topics")
    print(f"  q. Quit\n")

    while True:
        choice = input("Enter choice: ").strip().lower()
        if choice == "q":
            break
        elif choice == "a":
            for k in CONCEPTS:
                explore(k)
        elif choice in CONCEPTS:
            explore(choice)
        else:
            print("Invalid choice. Try again.")
