"""Replays a Scenario's timestamped transcript fragments through a live
CopilotBriefEngine, one chunk at a time — this IS the "incoming stream"
from the architecture diagram, just sourced from a fixture instead of a
live microphone/ASR feed. Used by both the CLI (run_demo.py) and the
FastAPI WebSocket demo endpoint so the two never diverge in behavior.
"""

from __future__ import annotations

import time
from typing import Iterator

from copilotbrief.models import TranscriptChunk
from copilotbrief.pipeline import CopilotBriefEngine

from .scenarios import Scenario


def _answer_dict(answer) -> dict | None:
    if answer is None:
        return None
    return {
        "version": answer.version,
        "text": answer.text,
        "citations": answer.citations,
        "uncertainty": answer.uncertainty,
        "changed_claim_ids": answer.changed_claim_ids,
    }


def replay_scenario(engine: CopilotBriefEngine, scenario: Scenario, speed: float = 0.0) -> Iterator[dict]:
    """Yields one event dict per transcript chunk, in real order across all
    turns of the scenario. `speed` scales real-time playback: 0 = instant
    (default, used by tests/benchmark), 1.0 = replay at the original
    inter-chunk timing, >1 = slower than real time (nice for a demo video)."""

    prev_ts = None
    for turn in scenario.turns:
        chunks = turn.chunks
        for i, (ts, text) in enumerate(chunks):
            if speed > 0 and prev_ts is not None:
                gap = max(0.0, ts - prev_ts)
                time.sleep(gap * speed)
            prev_ts = ts

            is_end = i == len(chunks) - 1
            chunk = TranscriptChunk(
                session_id=scenario.session_id, text=text, timestamp_s=ts, is_utterance_end=is_end,
            )
            result = engine.process_chunk(chunk)
            yield {
                "timestamp_s": ts,
                "chunk_text": text,
                "controller_decision": result.controller_decision.value,
                "controller_reason": result.controller_reason,
                "is_utterance_end": is_end,
                "answer": _answer_dict(result.answer),
            }
