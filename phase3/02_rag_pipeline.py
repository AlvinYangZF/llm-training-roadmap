"""Phase 3, Step 4: Simple RAG pipeline with local embeddings + FAISS.

Run: python3 phase3/02_rag_pipeline.py
Prerequisites: ollama pull nomic-embed-text
"""
import os
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Sample documents (replace with your own files later) ---
DOCUMENTS = [
    """KV Cache: During autoregressive generation, the transformer computes Key and Value
    matrices for each token. Without caching, every new token would require recomputing
    K and V for ALL previous tokens. The KV cache stores these computed K/V pairs so each
    new token only needs to compute its own K/V and attend to the cached values. This is
    why the first token is slow (prefill — process entire prompt) but subsequent tokens
    are fast (decode — only process one new token). The KV cache grows linearly with
    sequence length and number of layers.""",

    """Quantization reduces the precision of model weights from higher bit formats
    (FP32, FP16) to lower ones (INT8, INT4). A 7B parameter model in FP16 takes ~14GB
    of memory, but quantized to INT4 (Q4_0) it fits in ~4GB. The tradeoff is some loss
    in output quality, but modern quantization methods (GPTQ, AWQ, GGUF Q4_K_M) preserve
    most of the model's capabilities. This is what makes running LLMs on consumer hardware
    (like an 8GB MacBook) possible.""",

    """The Transformer architecture uses self-attention to process sequences. Each token
    creates Query (Q), Key (K), and Value (V) vectors. Attention scores are computed as
    softmax(QK^T / sqrt(d_k)) * V. Multi-head attention runs this in parallel across
    multiple heads, each learning different patterns. The architecture includes layer
    normalization, feed-forward networks, and residual connections. Modern LLMs typically
    use decoder-only transformers with causal (masked) attention.""",

    """GGUF (GPT-Generated Unified Format) is a file format for storing quantized LLM
    weights, developed for llama.cpp. It replaced the older GGML format and stores model
    architecture, tokenizer, and weights in a single file. Different quantization levels
    are denoted by names like Q4_0 (4-bit, basic), Q4_K_M (4-bit, better quality),
    Q5_K_M (5-bit), Q8_0 (8-bit). Ollama uses GGUF format internally.""",

    """Context Window refers to the maximum number of tokens an LLM can process at once.
    It includes both the input (prompt) and the output (generated text). Llama 3.2 has
    a 128K context window. Longer contexts require more memory (the KV cache grows) and
    more computation. Techniques like sliding window attention, RoPE scaling, and sparse
    attention help extend context lengths beyond training limits.""",
]


def build_vector_store() -> FAISS:
    """Split documents, embed with Ollama, store in FAISS."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents(DOCUMENTS)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def build_rag_chain(vectorstore: FAISS):
    """Create a RAG chain: retrieve → format → generate."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    llm = ChatOllama(model="llama3.2:3b")

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based on the following context. If the context doesn't
contain the answer, say so.

Context:
{context}

Question: {question}

Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


if __name__ == "__main__":
    print("Building vector store with local embeddings...")
    vs = build_vector_store()
    print("Done! Creating RAG chain...\n")

    chain = build_rag_chain(vs)

    questions = [
        "What is a KV cache and why does it make generation faster?",
        "How does quantization help run models on a MacBook?",
        "What is GGUF format?",
        "What determines the context window size?",
    ]

    for q in questions:
        print(f"Q: {q}")
        answer = chain.invoke(q)
        print(f"A: {answer}\n")
        print("-" * 60)
