"""Ephemeral, session-bound state store.

Per the spec's "Session-Bound State" hard rule: "Cross-session profiling
and persistent user tracking across independent test runs are prohibited.
Memory is strictly ephemeral and scoped to the active conversation
session." This store is a plain in-process dict — nothing is written to
disk, nothing survives process restart, and there is no cross-session
keying of any kind beyond the session_id the caller supplies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .models import AnswerVersion, Claim, RetrievalEvent, TelemetryEvent


@dataclass
class SessionState:
    session_id: str
    claims: dict[str, Claim] = field(default_factory=dict)
    active_claim_order: list[str] = field(default_factory=list)
    answer_versions: list[AnswerVersion] = field(default_factory=list)
    retrieval_events: list[RetrievalEvent] = field(default_factory=list)
    telemetry: list[TelemetryEvent] = field(default_factory=list)
    created_at_s: float = field(default_factory=time.time)
    last_active_s: float = field(default_factory=time.time)

    @property
    def current_version(self) -> int:
        return self.answer_versions[-1].version if self.answer_versions else 0

    @property
    def has_prior_answer(self) -> bool:
        return len(self.answer_versions) > 0

    def touch(self) -> None:
        self.last_active_s = time.time()

    def claims_snapshot(self) -> list[dict]:
        """JSON-friendly view of the currently active claim set, in display
        order, for driving UI intent cards / evidence panels — the raw
        `Claim` dataclasses aren't sent over the wire directly so the wire
        format can evolve independently of the internal model."""

        snapshot = []
        for claim_id in self.active_claim_order:
            claim = self.claims.get(claim_id)
            if claim is None:
                continue
            snapshot.append({
                "claim_id": claim.claim_id,
                "sub_query": claim.sub_query,
                "text": claim.text,
                "citations": claim.citations,
                "supported": claim.supported,
                "version": claim.version,
                "evidence": claim.evidence,
            })
        return snapshot


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get_or_create(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        state = self._sessions[session_id]
        state.touch()
        return state

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def all_session_ids(self) -> list[str]:
        return list(self._sessions.keys())
