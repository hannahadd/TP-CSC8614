import sys
import uuid
from typing import Any, List

try:
    from TP5.agent.tools.rag_tool import rag_search_tool as _rag_search
except Exception:
    from TP5.agent.tools.rag_tool import rag_search as _rag_search


def _call_rag(run_id: str, query: str, k: int) -> List[Any]:
    try:
        return _rag_search(run_id=run_id, query=query, k=k)
    except TypeError:
        return _rag_search(run_id, query, k)


def main() -> int:
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print('Usage: python TP5/rag_answer_tp5.py "votre question"')
        return 2

    run_id = str(uuid.uuid4())
    k = 5

    docs = _call_rag(run_id, query, k)

    print(f"run_id={run_id}")
    print(f"query={query}")
    print(f"k={k}")
    print("-" * 60)

    if not docs:
        print("No documents retrieved.")
        return 0

    for i, d in enumerate(docs, 1):
        doc_id = getattr(d, "doc_id", None) or (d.get("doc_id") if isinstance(d, dict) else None) or f"doc_{i}"
        text = getattr(d, "text", None) or (d.get("text") if isinstance(d, dict) else None) or str(d)
        text = (text or "").replace("\n", " ").strip()
        print(f"[{doc_id}] {text[:300]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
