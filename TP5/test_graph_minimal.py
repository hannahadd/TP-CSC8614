import uuid
from typing import Any, Dict

from TP5.load_test_emails import load_all_emails
from TP5.agent.state import AgentState
from TP5.agent.graph_minimal import build_graph


def _to_state(out: Any) -> AgentState:
    # LangGraph peut renvoyer soit AgentState, soit dict
    if isinstance(out, AgentState):
        return out
    if isinstance(out, dict):
        # Pydantic v2
        if hasattr(AgentState, "model_validate"):
            return AgentState.model_validate(out)
        # fallback
        return AgentState(**out)
    raise TypeError(f"Unexpected output type: {type(out)}")


def run_one_email(app, emails: list[Dict[str, Any]], email_id: str) -> AgentState:
    e = next(x for x in emails if x["email_id"] == email_id)

    state = AgentState(
        run_id=str(uuid.uuid4()),
        email_id=e["email_id"],
        subject=e["subject"],
        sender=e["from"],
        body=e["body"],
    )

    out = app.invoke(state)
    return _to_state(out)


if __name__ == "__main__":
    emails = load_all_emails()
    app = build_graph()

    # 1) Mode scan : trouve un escalate/ignore
    for email_id in ["E04"]:
        print("\n" + "=" * 70)
        print("EMAIL =", email_id)

        st = run_one_email(app, emails, email_id)

        print("\n=== DECISION ===")
        print("intent =", st.decision.intent)
        print("risk_level =", st.decision.risk_level)
        print("needs_retrieval =", st.decision.needs_retrieval)

        print("\n=== FINAL ===")
        print("kind =", st.final_kind)
        print(st.final_text)

        print("\n=== ACTIONS ===")
        print(st.actions)

        print("\nSUMMARY:", email_id, "->", st.decision.intent, "/", st.final_kind)
