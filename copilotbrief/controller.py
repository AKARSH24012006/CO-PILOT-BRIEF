"""Component 1 — Retrieval Controller.

Evaluates incoming transcript fragments to decide: WAIT for semantic
stabilization, fire a PROVISIONAL_RETRIEVE once a concrete entity has
stabilized (even if the utterance keeps going), escalate to
MULTI_INTENT_RETRIEVE once >=2 distinct sub-questions are distinguishable,
or return NO_RETRIEVAL for presentation-only turns (Example 3 in the spec).

This is the component the "Eager / Premature Retrieval on Noise" and
"Ignoring Presentation-Only Turns" pitfalls in the Theme 4 guide are aimed
at, so both are handled explicitly below rather than left implicit.
"""

from __future__ import annotations

import re

from . import clausing, config, taxonomy
from .models import ControllerDecision, ControllerVerdict, TranscriptChunk
from .retrieval import tokenize

_PRESENTATION_ONLY_RE = re.compile(
    r"\b(repeat|rephrase|restate|reformat|re-?word|summari[sz]e|shorten|"
    r"translate|format)\b.*\b(that|it|your|last|previous|answer|response)\b"
    r"|\b(in|as)\s+(two|three|four|a\s+few|bullet|bullets|a\s+list|one\s+sentence)\b"
    r"|\bmake\s+it\s+(shorter|more\s+concise|a\s+list|bullets?)\b",
    re.IGNORECASE,
)


def _ends_dangling(text: str) -> bool:
    words = tokenize(text)
    if not words:
        return True
    return words[-1] in config.DANGLING_CONNECTIVES


def _looks_like_multi_intent(text: str) -> int:
    """Count of distinct, content-bearing clauses (shared logic with
    decomposer.py via clausing.py, so the controller's clause count and
    the decomposer's actual sub-query count never disagree)."""

    return max(1, len(clausing.split_clauses(text, min_tokens=config.MIN_SUBQUERY_TOKENS)))


class RetrievalController:
    """Stateful across a single session's utterance stream.

    One instance is meant to live for the lifetime of a conversation
    session; `reset_utterance` should be called whenever a new user turn
    begins (session-bound, ephemeral — no cross-session state per the
    spec's "Session-Bound State" rule).
    """

    def __init__(self) -> None:
        self._state: dict[str, dict] = {}

    def _get_state(self, session_id: str) -> dict:
        return self._state.setdefault(
            session_id,
            {
                "accumulated": "",
                "provisional_fired": False,
                "multi_intent_fired": False,
                "last_clause_count": 1,
            },
        )

    def reset_utterance(self, session_id: str) -> None:
        self._state[session_id] = {
            "accumulated": "",
            "provisional_fired": False,
            "multi_intent_fired": False,
            "last_clause_count": 1,
        }

    def evaluate(self, chunk: TranscriptChunk, has_prior_answer: bool = False) -> ControllerVerdict:
        state = self._get_state(chunk.session_id)
        accumulated = (state["accumulated"] + " " + chunk.text).strip()
        state["accumulated"] = accumulated

        # --- No-Retrieval: presentation-only turn on existing context -----
        if has_prior_answer and _PRESENTATION_ONLY_RE.search(chunk.text):
            return ControllerVerdict(
                decision=ControllerDecision.NO_RETRIEVAL,
                reason="presentation_restructure",
                accumulated_text=accumulated,
                is_utterance_end=chunk.is_utterance_end,
            )

        clause_count = _looks_like_multi_intent(accumulated)

        # --- Multi-Intent Retrieve: >=2 distinct clauses detected ----------
        if clause_count >= 2 and (not state["multi_intent_fired"] or clause_count > state["last_clause_count"]):
            state["multi_intent_fired"] = True
            state["provisional_fired"] = True  # multi-intent supersedes a bare provisional fire
            state["last_clause_count"] = clause_count
            return ControllerVerdict(
                decision=ControllerDecision.MULTI_INTENT_RETRIEVE,
                reason=f"decompose_{clause_count}_subintents",
                accumulated_text=accumulated,
                is_utterance_end=chunk.is_utterance_end,
            )

        # --- Provisional Retrieve: a stable entity has appeared -------------
        stable_tokens = len(tokenize(accumulated))
        if (
            not state["provisional_fired"]
            and stable_tokens >= config.MIN_STABLE_TOKENS_FOR_PROVISIONAL
            and taxonomy.has_stable_entity(accumulated)
        ):
            state["provisional_fired"] = True
            return ControllerVerdict(
                decision=ControllerDecision.PROVISIONAL_RETRIEVE,
                reason="stable_entity_identified",
                accumulated_text=accumulated,
                is_utterance_end=chunk.is_utterance_end,
            )

        # --- Otherwise: keep waiting ----------------------------------------
        reason = "incomplete_semantically_unstable" if _ends_dangling(accumulated) else "awaiting_stable_entity"
        return ControllerVerdict(
            decision=ControllerDecision.WAIT,
            reason=reason,
            accumulated_text=accumulated,
            is_utterance_end=chunk.is_utterance_end,
        )
