# TP5/agent/nodes/finalize.py
import re
from typing import List

from TP5.agent.logger import log_event
from TP5.agent.state import AgentState

RE_CIT = re.compile(r"\[(doc_\d+)\]")

def _extract_citations(text: str) -> List[str]:
    return sorted(set(RE_CIT.findall(text or "")))

def _fallback_reply() -> str:
    return (
        "Bonjour,\n\n"
        "Merci pour votre message. Je reviens vers vous dès que possible.\n\n"
        "Cordialement,"
    )

def _fallback_questions() -> str:
    # 1–3 questions, actionnables
    return (
        "Pouvez-vous préciser le contexte exact (service/UE/procédure) ?\n"
        "Quelle est la demande précise et l’échéance ?\n"
        "Avez-vous un lien, un formulaire ou une référence associée ?"
    )

def _build_handoff_summary(state: AgentState) -> str:
    d = getattr(state, "decision", None)
    parts = []
    if d is not None:
        intent = getattr(d, "intent", "")
        if intent:
            parts.append(f"intent={intent}")
        rq = getattr(d, "retrieval_query", "")
        if rq:
            parts.append(f"retrieval_query={rq}")
        rat = getattr(d, "rationale", "")
        if rat:
            parts.append(f"rationale={rat}")

    dv1 = (getattr(state, "draft_v1", "") or "").strip()
    if dv1:
        parts.append(f"draft_v1={dv1[:300]}")  # court et traçable

    return " | ".join(parts)[:900] or "Escalade requise (résumé indisponible)."

def finalize(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "finalize"})
    if not state.budget.can_step():
        log_event(state.run_id, "node_end", {"node": "finalize", "status": "budget_exceeded"})
        return state

    state.budget.steps_used += 1


    intent = getattr(state.decision, "intent", "")

    if intent == "reply":
        state.final_kind = "reply"
        base = (state.draft_v1 or "").strip() or _fallback_reply()

        # citations “détectables” : d’abord dans le texte, sinon via evidence.doc_id si dispo
        cits = _extract_citations(base)
        if not cits and getattr(state, "evidence", None):
            cits = sorted({getattr(d, "doc_id", "") for d in state.evidence if getattr(d, "doc_id", "")})

        if cits:
            state.final_text = base + "\n\nSources: " + " ".join(f"[{c}]" for c in cits)
        else:
            state.final_text = base

    elif intent == "ask_clarification":
        state.final_kind = "clarification"
        state.final_text = (state.draft_v1 or "").strip() or _fallback_questions()

    elif intent == "escalate":
        state.final_kind = "handoff"
        summary = _build_handoff_summary(state)
        evidence_ids = [getattr(d, "doc_id", "") for d in (state.evidence or []) if getattr(d, "doc_id", "")]

        # action mockée (packet)
        state.actions.append({
            "type": "handoff_packet",
            "run_id": state.run_id,
            "email_id": state.email_id,
            "summary": summary,
            "evidence_ids": evidence_ids,
        })

        state.final_text = "Votre demande nécessite une validation humaine. Je transmets avec un résumé et les sources disponibles."

    else:
        state.final_kind = "ignore"
        state.final_text = "Ignoré : non actionnable / hors périmètre."

    log_event(state.run_id, "node_end", {"node": "finalize", "status": "ok", "final_kind": state.final_kind})
    return state
