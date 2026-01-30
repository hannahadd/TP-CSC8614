import os
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from TP5.agent.logger import log_event
from TP5.agent.state import EvidenceDoc

# Consigne: Chroma depuis TP5/chroma_db
#CHROMA_DIR = os.path.join("TP5", "chroma_db")

# IMPORTANT: même nom que TP précédent (TP4)
# Astuce: grep -R "COLLECTION_NAME" -n TP4 | head
#COLLECTION_NAME = os.environ.get("TP5_COLLECTION_NAME", "emails")

# IMPORTANT: même embedding model que TP précédent
#EMBEDDING_MODEL = os.environ.get("TP5_EMBEDDING_MODEL", "bge-m3:latest")

# Ollama (modifiable)
#PORT = os.environ.get("OLLAMA_PORT", "11434")

CHROMA_DIR = os.path.join("TP5", "chroma_db")
COLLECTION_NAME = "empty_for_tp7"
#COLLECTION_NAME = "tp4_rag"          # même nom que TP4
EMBEDDING_MODEL = "bge-m3:latest"    # même embedding que TP4
PORT = "11434"                       # ou ton port (ex: 11435)

def _hash_args(args: Dict[str, Any]) -> str:
    raw = repr(sorted(args.items())).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _format_snippet(doc: Document, max_len: int = 320) -> str:
    txt = (doc.page_content or "").strip().replace("\n", " ")
    return (txt[:max_len] + "...") if len(txt) > max_len else txt


def rag_search_tool(
    run_id: str,
    query: str,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None
) -> List[EvidenceDoc]:
    """
    Tool RAG : retourne des EvidenceDoc citables.
    """
    filters = filters or {}
    t0 = time.time()
    args = {"query": query, "k": k, "filters": filters}
    args_hash = _hash_args(args)

    if (not query.strip()) or (k > 10):
        log_event(run_id, "tool_call", {
            "tool": "rag_search",
            "args_hash": args_hash,
            "latency_ms": int((time.time() - t0) * 1000),
            "status": "error",
            "error": "invalid_args"
        })
        return []

    try:
        emb = OllamaEmbeddings(
            base_url=f"http://127.0.0.1:{PORT}",
            model=EMBEDDING_MODEL,
        )

        vectordb = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=emb,
            persist_directory=CHROMA_DIR,
        )

        evidence: List[EvidenceDoc] = []

        # On essaie d'avoir un score (relevance score), sinon fallback sans score
        docs_scores: List[Tuple[Document, float]] = []
        try:
            docs_scores = vectordb.similarity_search_with_relevance_scores(
                query, k=k, filter=(filters or None)
            )
        except Exception:
            retriever_kwargs = {"k": k}
            if filters:
                retriever_kwargs["filter"] = filters
            retriever = vectordb.as_retriever(search_kwargs=retriever_kwargs)
            docs = retriever.invoke(query)
            docs_scores = [(d, None) for d in docs]  # type: ignore

        for i, (d, score) in enumerate(docs_scores, start=1):
            meta = d.metadata or {}
            evidence.append(EvidenceDoc(
                doc_id=f"doc_{i}",
                doc_type=str(meta.get("doc_type", "unknown")),
                source=str(meta.get("source", meta.get("filename", "unknown"))),
                snippet=_format_snippet(d),
                score=score if isinstance(score, (float, int)) else None,
            ))

        log_event(run_id, "tool_call", {
            "tool": "rag_search",
            "args_hash": args_hash,
            "latency_ms": int((time.time() - t0) * 1000),
            "status": "ok",
            "k": k,
            "n_docs": len(evidence),
        })
        return evidence

    except Exception as e:
        log_event(run_id, "tool_call", {
            "tool": "rag_search",
            "args_hash": args_hash,
            "latency_ms": int((time.time() - t0) * 1000),
            "status": "error",
            "error": str(e),
        })
        return []
