"""Phase 3, Step 3: Build a simple LangChain chain with local Ollama model.

Run: python3 phase3/01_langchain_chain.py
"""
import os
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Connect to local Ollama
llm = ChatOllama(model="llama3.2:3b")

# Simple chain: prompt template → LLM → string output
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} simply in 3 sentences. Use an analogy if helpful."
)
chain = prompt | llm | StrOutputParser()


def explain(topic: str) -> str:
    return chain.invoke({"topic": topic})


# Multi-step chain: generate then critique
critic_prompt = ChatPromptTemplate.from_template(
    "Rate this explanation on clarity (1-10) and suggest one improvement:\n\n{explanation}"
)
critique_chain = prompt | llm | StrOutputParser() | (lambda text: {"explanation": text}) | critic_prompt | llm | StrOutputParser()


if __name__ == "__main__":
    print("=== Simple Chain ===")
    topics = ["KV cache", "quantization", "attention mechanism"]
    for t in topics:
        print(f"\n--- {t} ---")
        print(explain(t))

    print("\n=== Chain with Self-Critique ===")
    result = critique_chain.invoke({"topic": "KV cache"})
    print(result)
