"""Component 5 — Observability / Telemetry.

Emits structured, timestamped events for every controller decision,
retrieval call, and synthesis/refinement step, and can assemble them into
the "Structured Output Event Record" shape shown in the Theme 4 guide
(retrieval_events, sub_queries, answer, citations, uncertainty) plus the
extra fields Gate G6 asks for: answer version lineage and a token-cost
estimate.

100% trace coverage (Gate G6) means every pipeline step logs something —
so every call site in pipeline.py logs through this module, never silently.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Callable, Optional

from .models import RetrievalEvent, TelemetryEvent
from .session import SessionState

Sink = Callable[[TelemetryEvent], None]


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~0.75 words/token for English)."""
    words = len(text.split())
    return max(1, round(words / 0.75))


class TelemetryLogger:
    def __init__(self, sink: Optional[Sink] = None) -> None:
        self._sink = sink

    def set_sink(self, sink: Optional[Sink]) -> None:
        self._sink = sink

    def log(self, session: SessionState, event_type: str, payload: dict, timestamp_s: Optional[float] = None) -> TelemetryEvent:
        event = TelemetryEvent(
            event_type=event_type,
            timestamp_s=timestamp_s if timestamp_s is not None else time.time(),
            payload=payload,
        )
        session.telemetry.append(event)
        if self._sink is not None:
            try:
                self._sink(event)
            except Exception:
                pass  # telemetry delivery must never break the pipeline
        return event

    @staticmethod
    def to_jsonl(session: SessionState) -> str:
        lines = []
        for e in session.telemetry:
            lines.append(json.dumps({
                "event_id": e.event_id,
                "event_type": e.event_type,
                "timestamp_s": round(e.timestamp_s, 3) if e.timestamp_s > 1e6 else e.timestamp_s,
                "payload": e.payload,
            }))
        return "\n".join(lines)

    @staticmethod
    def structured_output_record(session: SessionState) -> dict:
        """Assemble the spec's "Structured Output Event Record" shape from
        current session state: retrieval_events, sub_queries, answer,
        citations, uncertainty — plus version lineage and token cost."""

        latest = session.answer_versions[-1] if session.answer_versions else None
        retrieval_events = [
            {
                "timestamp_s": re.timestamp_s,
                "query": re.query,
                "trigger": re.trigger,
                "result_count": re.result_count,
            }
            for re in session.retrieval_events
        ]
        sub_queries = sorted({c.sub_query for c in session.claims.values()})
        total_tokens = sum(estimate_tokens(v.text) for v in session.answer_versions)

        return {
            "session_id": session.session_id,
            "retrieval_events": retrieval_events,
            "sub_queries": sub_queries,
            "answer": latest.text if latest else None,
            "citations": latest.citations if latest else [],
            "uncertainty": latest.uncertainty if latest else None,
            "answer_version": latest.version if latest else 0,
            "answer_version_lineage": [
                {"version": v.version, "changed_claims": v.changed_claim_ids, "created_at_s": v.created_at_s}
                for v in session.answer_versions
            ],
            "token_cost_estimate": total_tokens,
        }
