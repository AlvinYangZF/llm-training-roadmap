"""
Local RAG Agent v2 — LangChain + Ollama + FAISS + Hybrid Search + Reranking

Optimized RAG agent with P0 improvements:
1. Loads documents from a local directory (PDF, MD, TXT, DOCX)
2. Optimized chunking (512 tokens, 64 overlap) for better retrieval precision
3. BGE-M3 embeddings (upgraded from nomic-embed-text) for higher quality vectors
4. Hybrid search: BM25 (sparse) + FAISS (dense) ensemble for best of both worlds
5. FlashRank reranking: retrieve top-20, rerank to top-5 for precision
6. Conversational memory with source attribution
7. Context reordering to mitigate "lost in the middle" problem

Usage:
    # Index documents and start interactive chat
    python rag_agent/agent.py --docs ./docs

    # Query directly from command line
    python rag_agent/agent.py --docs ./docs --query "What is PagedAttention?"

    # Use a different LLM model
    python rag_agent/agent.py --docs ./docs --model deepseek-r1:8b

    # Use legacy embedding model (nomic-embed-text)
    python rag_agent/agent.py --docs ./docs --embed-model nomic-embed-text

    # Disable reranking for faster (but less precise) results
    python rag_agent/agent.py --docs ./docs --no-rerank
"""
import os
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import argparse
import glob
import hashlib
import json
import time
from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)


class SimpleMarkdownLoader:
    """Simple markdown loader that doesn't require unstructured."""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        from langchain_core.documents import Document
        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return [Document(page_content=text, metadata={"source": self.file_path})]


# --- Configuration ---
DEFAULT_MODEL = "llama3.2:3b"
EMBED_MODEL = "bge-m3"              # Upgraded: higher MTEB score, better multilingual
EMBED_MODEL_FALLBACK = "nomic-embed-text"
CHUNK_SIZE = 512                     # Optimized: was 800, 512 is benchmark-validated sweet spot
CHUNK_OVERLAP = 64                   # Optimized: was 100, ~12% overlap
RETRIEVE_K = 20                      # Retrieve more candidates for reranking
RERANK_TOP_N = 5                     # Rerank down to top 5 for precision
BM25_WEIGHT = 0.4                    # BM25 weight in hybrid search
DENSE_WEIGHT = 0.6                   # Dense (FAISS) weight in hybrid search
INDEX_DIR = ".rag_index_v2"          # New index dir (v2 uses different chunking/embedding)


# --- Document Loading ---
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".md": SimpleMarkdownLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
}


def load_documents(docs_dir: str):
    """Load all supported documents from a directory."""
    docs = []
    supported = list(LOADER_MAP.keys())
    files = []
    for ext in supported:
        files.extend(glob.glob(os.path.join(docs_dir, f"**/*{ext}"), recursive=True))

    if not files:
        print(f"  No supported files found in {docs_dir}")
        print(f"  Supported formats: {', '.join(supported)}")
        return docs

    for fpath in sorted(files):
        ext = Path(fpath).suffix.lower()
        loader_cls = LOADER_MAP.get(ext)
        if not loader_cls:
            continue
        try:
            loader = loader_cls(fpath)
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = os.path.relpath(fpath, docs_dir)
            docs.extend(loaded)
            print(f"  Loaded: {os.path.relpath(fpath, docs_dir)} ({len(loaded)} chunks)")
        except Exception as e:
            print(f"  Skipped: {os.path.relpath(fpath, docs_dir)} ({e})")
    return docs


def compute_docs_hash(docs_dir: str, embed_model: str) -> str:
    """Compute a hash of all document files + config for cache invalidation."""
    h = hashlib.md5()
    # Include config in hash so changing settings triggers rebuild
    h.update(f"embed={embed_model},chunk={CHUNK_SIZE},overlap={CHUNK_OVERLAP}".encode())
    supported = list(LOADER_MAP.keys())
    files = []
    for ext in supported:
        files.extend(glob.glob(os.path.join(docs_dir, f"**/*{ext}"), recursive=True))
    for fpath in sorted(files):
        h.update(fpath.encode())
        h.update(str(os.path.getmtime(fpath)).encode())
        h.update(str(os.path.getsize(fpath)).encode())
    return h.hexdigest()


def _check_ollama_model(model_name: str) -> bool:
    """Check if an Ollama model is available locally."""
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        return model_name in result.stdout
    except Exception:
        return False


def _resolve_embed_model(requested: str) -> str:
    """Resolve embedding model, falling back if requested isn't available."""
    if _check_ollama_model(requested):
        return requested
    if requested != EMBED_MODEL_FALLBACK and _check_ollama_model(EMBED_MODEL_FALLBACK):
        print(f"  Warning: {requested} not found, falling back to {EMBED_MODEL_FALLBACK}")
        return EMBED_MODEL_FALLBACK
    return requested  # Let it fail with a clear error if neither exists


# --- Vector Store ---
def build_vector_store(docs_dir: str, embed_model: str, force_rebuild: bool = False):
    """Build or load a cached FAISS vector store. Returns (vectorstore, chunks)."""
    index_path = os.path.join(docs_dir, INDEX_DIR)
    hash_file = os.path.join(index_path, "docs_hash.json")
    chunks_file = os.path.join(index_path, "chunks.json")
    current_hash = compute_docs_hash(docs_dir, embed_model)

    embeddings = OllamaEmbeddings(model=embed_model)

    # Try loading cached index
    if not force_rebuild and os.path.exists(os.path.join(index_path, "index.faiss")):
        if os.path.exists(hash_file):
            with open(hash_file) as f:
                cached_hash = json.load(f).get("hash")
            if cached_hash == current_hash:
                print("Loading cached vector store...")
                vs = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
                print(f"  Loaded {vs.index.ntotal} vectors from cache")
                # Load cached chunks for BM25
                chunks = _load_cached_chunks(chunks_file)
                print(f"  Loaded {len(chunks)} chunks for BM25\n")
                return vs, chunks
        print("Documents changed, rebuilding index...")
    else:
        print("Building vector store...")

    # Load and split documents
    docs = load_documents(docs_dir)
    if not docs:
        raise ValueError(f"No documents found in {docs_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"\n  Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # Embed and store
    print(f"  Embedding with {embed_model}...")
    t0 = time.time()
    vs = FAISS.from_documents(chunks, embeddings)
    elapsed = time.time() - t0
    print(f"  Embedded {len(chunks)} chunks in {elapsed:.1f}s")

    # Cache the index
    os.makedirs(index_path, exist_ok=True)
    vs.save_local(index_path)
    with open(hash_file, "w") as f:
        json.dump({"hash": current_hash}, f)
    # Cache chunks for BM25 (text + metadata only)
    _save_chunks_cache(chunks, chunks_file)
    print(f"  Saved index to {index_path}\n")

    return vs, chunks


def _save_chunks_cache(chunks, path: str):
    """Save chunk text/metadata for BM25 retriever reconstruction."""
    data = [{"text": c.page_content, "metadata": c.metadata} for c in chunks]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _load_cached_chunks(path: str):
    """Load cached chunks for BM25."""
    from langchain_core.documents import Document
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Document(page_content=d["text"], metadata=d["metadata"]) for d in data]


# --- Reciprocal Rank Fusion ---
def reciprocal_rank_fusion(doc_lists, weights, k=60):
    """Merge multiple ranked doc lists using weighted Reciprocal Rank Fusion.

    RRF score = sum(weight_i / (k + rank_i)) for each retriever that returned the doc.
    """
    fused_scores = {}  # doc content -> (score, doc)
    for doc_list, weight in zip(doc_lists, weights):
        for rank, doc in enumerate(doc_list):
            doc_key = doc.page_content  # strings are hashable; avoids hash() randomization
            if doc_key not in fused_scores:
                fused_scores[doc_key] = (0.0, doc)
            prev_score, _ = fused_scores[doc_key]
            fused_scores[doc_key] = (prev_score + weight / (k + rank + 1), doc)

    # Sort by fused score descending
    sorted_docs = sorted(fused_scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in sorted_docs]


class HybridRetriever:
    """BM25 + FAISS hybrid retriever with RRF fusion."""

    def __init__(self, faiss_retriever, bm25_retriever, weights, top_k):
        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever
        self.weights = weights
        self.top_k = top_k

    def invoke(self, query, **kwargs):
        faiss_docs = self.faiss_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        fused = reciprocal_rank_fusion(
            [bm25_docs, faiss_docs], self.weights
        )
        return fused[:self.top_k]


class RerankingRetriever:
    """Wraps a base retriever with FlashRank reranking."""

    def __init__(self, base_retriever, top_n):
        from flashrank import Ranker, RerankRequest
        self.base_retriever = base_retriever
        self.top_n = top_n
        self.ranker = Ranker()
        self._RerankRequest = RerankRequest

    def invoke(self, query, **kwargs):
        docs = self.base_retriever.invoke(query)
        if not docs:
            return docs
        try:
            passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
            rerank_request = self._RerankRequest(query=query, passages=passages)
            results = self.ranker.rerank(rerank_request)
            # Map back to LangChain docs — handle both dict and object result formats
            reranked = []
            for r in results[:self.top_n]:
                idx = r.get("id") if isinstance(r, dict) else getattr(r, "id", None)
                if idx is not None and 0 <= idx < len(docs):
                    reranked.append(docs[idx])
            return reranked if reranked else docs[:self.top_n]
        except Exception as e:
            print(f"  Warning: reranking failed ({e}), using unranked results")
            return docs[:self.top_n]


# --- Hybrid Retriever (BM25 + FAISS + Reranking) ---
def build_hybrid_retriever(vs: FAISS, chunks, use_rerank: bool = True):
    """Build a hybrid retriever: BM25 + FAISS ensemble, optionally with FlashRank reranking."""
    from langchain_community.retrievers import BM25Retriever

    # Dense retriever (FAISS)
    faiss_retriever = vs.as_retriever(search_kwargs={"k": RETRIEVE_K})

    # Sparse retriever (BM25)
    bm25_retriever = BM25Retriever.from_documents(chunks, k=RETRIEVE_K)

    # Hybrid ensemble with Reciprocal Rank Fusion
    hybrid = HybridRetriever(
        faiss_retriever=faiss_retriever,
        bm25_retriever=bm25_retriever,
        weights=[BM25_WEIGHT, DENSE_WEIGHT],
        top_k=RETRIEVE_K,
    )

    if not use_rerank:
        print(f"  Hybrid search: BM25({BM25_WEIGHT}) + FAISS({DENSE_WEIGHT}), top-{RETRIEVE_K}")
        return hybrid

    # Add FlashRank reranking on top
    try:
        reranking_retriever = RerankingRetriever(base_retriever=hybrid, top_n=RERANK_TOP_N)
        print(f"  Hybrid search: BM25({BM25_WEIGHT}) + FAISS({DENSE_WEIGHT}), "
              f"retrieve top-{RETRIEVE_K} → rerank to top-{RERANK_TOP_N}")
        return reranking_retriever
    except ImportError:
        print("  Warning: flashrank not installed, skipping reranking")
        print("  Install with: pip install flashrank")
        return hybrid


# --- Context Reordering ---
def reorder_docs(docs):
    """Reorder documents to mitigate 'lost in the middle' problem.

    Places the most relevant docs at the beginning and end of the context,
    since LLMs tend to ignore information in the middle of long contexts.
    Even-ranked (0, 2, 4...) go to the front, odd-ranked (1, 3, 5...) go to the back.
    """
    if len(docs) <= 2:
        return docs
    reordered = []
    for i, doc in enumerate(docs):
        if i % 2 == 0:
            reordered.insert(0, doc)  # most relevant docs go to FRONT
        else:
            reordered.append(doc)     # next-best go to BACK
    return reordered


# --- RAG Chain ---
def build_rag_chain(retriever, model: str = DEFAULT_MODEL):
    """Build a conversational RAG chain with source attribution."""
    llm = ChatOllama(model=model, temperature=0.1)

    system_prompt = """You are a helpful AI assistant that answers questions based on the provided context documents.

Rules:
1. Answer ONLY based on the provided context. If the context doesn't contain enough information, say so clearly.
2. Cite your sources by mentioning the document name when possible.
3. Be concise but thorough.
4. If the user asks a follow-up question, use the conversation history for context.

Context from documents:
{context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"[Source {i}: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def retrieve_and_format(input_dict):
        question = input_dict["question"]
        docs = retriever.invoke(question)
        # Apply context reordering
        docs = reorder_docs(docs)
        return {
            "context": format_docs(docs),
            "question": question,
            "chat_history": input_dict.get("chat_history", []),
            "source_docs": docs,
        }

    answer_chain = prompt | llm | StrOutputParser()

    def run_chain(input_dict):
        enriched = retrieve_and_format(input_dict)
        answer = answer_chain.invoke({
            "context": enriched["context"],
            "question": enriched["question"],
            "chat_history": enriched["chat_history"],
        })
        return {"answer": answer, "source_docs": enriched["source_docs"]}

    return RunnableLambda(run_chain)


# --- Interactive Chat ---
def interactive_chat(chain, verbose: bool = False):
    """Run an interactive chat session with the RAG agent."""
    print("=" * 60)
    print("  Local RAG Agent v2 — Chat with your documents")
    print("  Hybrid Search + Reranking + Context Reordering")
    print("=" * 60)
    print("  Commands: 'quit' to exit, 'clear' to reset history")
    print("=" * 60)
    print()

    chat_history = []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if question.lower() == "clear":
            chat_history.clear()
            print("Chat history cleared.\n")
            continue

        t0 = time.time()
        result = chain.invoke({
            "question": question,
            "chat_history": chat_history,
        })
        elapsed = time.time() - t0

        answer = result["answer"]
        sources = result["source_docs"]

        print(f"\nAssistant: {answer}")
        print(f"\n  Sources ({elapsed:.1f}s):")
        seen = set()
        for doc in sources:
            src = doc.metadata.get("source", "unknown")
            if src not in seen:
                seen.add(src)
                print(f"    - {src}")
        print()

        # Update conversation history
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))

        # Keep history manageable (last 10 turns)
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]


def single_query(chain, question: str):
    """Run a single query and print the result."""
    t0 = time.time()
    result = chain.invoke({
        "question": question,
        "chat_history": [],
    })
    elapsed = time.time() - t0

    print(f"\nQ: {question}")
    print(f"\nA: {result['answer']}")
    print(f"\nSources ({elapsed:.1f}s):")
    seen = set()
    for doc in result["source_docs"]:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            print(f"  - {src}")
    print()


# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Local RAG Agent v2 — LangChain + Ollama + Hybrid Search")
    parser.add_argument("--docs", required=True, help="Path to documents directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--embed-model", default=EMBED_MODEL, help=f"Embedding model (default: {EMBED_MODEL})")
    parser.add_argument("--query", "-q", help="Single query (skip interactive mode)")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild the vector index")
    parser.add_argument("--no-rerank", action="store_true", help="Disable FlashRank reranking")
    args = parser.parse_args()

    # Verify docs directory
    if not os.path.isdir(args.docs):
        print(f"Error: {args.docs} is not a directory")
        return

    # Resolve embedding model (with fallback)
    embed_model = _resolve_embed_model(args.embed_model)
    print(f"Embedding model: {embed_model}")

    # Build vector store
    vs, chunks = build_vector_store(args.docs, embed_model, force_rebuild=args.rebuild)

    # Build hybrid retriever
    print("Building hybrid retriever...")
    retriever = build_hybrid_retriever(vs, chunks, use_rerank=not args.no_rerank)

    # Build RAG chain
    print(f"LLM model: {args.model}\n")
    chain = build_rag_chain(retriever, model=args.model)

    # Run
    if args.query:
        single_query(chain, args.query)
    else:
        interactive_chat(chain)


if __name__ == "__main__":
    main()
