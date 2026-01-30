# TP5/run_batch.py
import os
import uuid
from typing import Any, List, Dict

from TP5.load_test_emails import load_all_emails
from TP5.agent.state import AgentState
from TP5.agent.graph_minimal import build_graph

OUT_MD = os.path.join("TP5", "batch_results.md")
RUNS_DIR = os.path.join("TP5", "runs")


def md_escape(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ")


def _to_state(out: Any) -> AgentState:
    # LangGraph peut renvoyer soit AgentState, soit dict
    if isinstance(out, AgentState):
        return out
    if isinstance(out, dict):
        # Pydantic v2
        if hasattr(AgentState, "model_validate"):
            return AgentState.model_validate(out)
        return AgentState(**out)
    raise TypeError(f"Unexpected output type: {type(out)}")


def _budget_int(st: AgentState, field: str, default: int = 0) -> int:
    b = getattr(st, "budget", None)
    if b is None:
        return default
    v = getattr(b, field, None)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _count_tool_calls_from_log(run_id: str) -> int:
    # fallback robuste : compter les lignes tool_call dans le jsonl du run
    path = os.path.join(RUNS_DIR, f"{run_id}.jsonl")
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if '"event": "tool_call"' in line:
                n += 1
    return n


def main():
    emails = load_all_emails()

    # Consigne: 8–12 emails (si tu en as plus, on limite à 12)
    emails = emails[:12]

    app = build_graph()

    rows: List[str] = []
    rows.append("| email_id | subject | intent | category | risk | final_kind | tool_calls | retrieval_attempts | notes |")
    rows.append("|---|---|---|---|---|---|---:|---:|---|")

    for e in emails:
        run_id = str(uuid.uuid4())
        state = AgentState(
            run_id=run_id,
            email_id=e["email_id"],
            subject=e["subject"],
            sender=e["from"],
            body=e["body"],
        )

        out = app.invoke(state)
        st = _to_state(out)

        intent = st.decision.intent
        category = st.decision.category
        risk = st.decision.risk_level
        final_kind = st.final_kind

        # tool_calls: d'abord via budget si dispo, sinon via le log jsonl
        tool_calls = _budget_int(st, "tool_calls_used", 0)
        if tool_calls == 0:
            tool_calls = _count_tool_calls_from_log(run_id)

        retrieval_attempts = _budget_int(st, "retrieval_attempts", 0)

        notes = f"log=TP5/runs/{run_id}.jsonl"

        rows.append(
            "| "
            + " | ".join([
                md_escape(st.email_id),
                md_escape(st.subject)[:60],
                intent,
                category,
                risk,
                final_kind,
                str(tool_calls),
                str(retrieval_attempts),
                md_escape(notes),
            ])
            + " |"
        )

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
