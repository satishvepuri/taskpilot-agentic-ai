"""
Document retrieval tool (RAG) — searches a small local knowledge base
(policies, FAQs, product docs) using embeddings + FAISS.

Kept intentionally simple (single FAISS index, no reranking) since the
point of this project is agentic orchestration, not RAG sophistication —
that depth already lives in the separate RAG project on the resume.
"""
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from environment

INDEX_DIR = Path(__file__).parent.parent.parent / "data" / "index"
EMBEDDING_MODEL = "text-embedding-3-small"


def embed_texts(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype="float32")


def build_index(documents: list[dict]) -> None:
    """
    documents: list of {"id": str, "text": str, "source": str}
    Builds and persists a FAISS index + metadata sidecar. Run once (or on
    doc updates) via `python -m src.tools.retrieval_tool` — not on every query.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    texts = [d["text"] for d in documents]
    embeddings = embed_texts(texts)

    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine sim via normalized inner product
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "docs.index"))
    with open(INDEX_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(documents, f)


def retrieve_docs(tool_input: str, top_k: int = 3) -> dict:
    """tool_input: the natural-language query to search the knowledge base with."""
    index_path = INDEX_DIR / "docs.index"
    meta_path = INDEX_DIR / "metadata.pkl"

    if not index_path.exists():
        raise RuntimeError(
            "No index found. Run `python -m src.tools.retrieval_tool` to build it first."
        )

    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        documents = pickle.load(f)

    query_embedding = embed_texts([tool_input])
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        doc = documents[idx]
        results.append({"source": doc["source"], "text": doc["text"], "score": float(score)})

    return {"query": tool_input, "results": results}


if __name__ == "__main__":
    # Sample knowledge base for demo purposes
    sample_docs = [
        {"id": "1", "source": "refund_policy.md", "text": "Refunds are issued within 5-7 business days for orders cancelled within 30 days of purchase."},
        {"id": "2", "source": "shipping_policy.md", "text": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days for an added fee."},
        {"id": "3", "source": "campaign_faq.md", "text": "Campaign budgets can be adjusted mid-flight but changes take up to 2 hours to propagate across ad exchanges."},
        {"id": "4", "source": "account_faq.md", "text": "Customers can update their billing information from the account settings page under 'Payment Methods'."},
    ]
    print("Building FAISS index from sample docs (requires OPENAI_API_KEY)...")
    build_index(sample_docs)
    print(f"Index built at {INDEX_DIR}")
