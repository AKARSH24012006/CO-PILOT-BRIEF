"""Shared dataclasses used across the CopilotBrief pipeline."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# --------------------------------------------------------------------------
# Streaming input
# --------------------------------------------------------------------------

@dataclass
class TranscriptChunk:
    """One incremental fragment of a streamed user utterance."""

    session_id: str
    text: str
    timestamp_s: float
    is_utterance_end: bool = False


# --------------------------------------------------------------------------
# Controller decisions (Component 1)
# --------------------------------------------------------------------------

class ControllerDecision(str, Enum):
    WAIT = "wait"
    PROVISIONAL_RETRIEVE = "provisional_retrieve"
    MULTI_INTENT_RETRIEVE = "multi_intent_retrieve"
    NO_RETRIEVAL = "no_retrieval"


@dataclass
class ControllerVerdict:
    decision: ControllerDecision
    reason: str
    accumulated_text: str
    is_utterance_end: bool = False


# --------------------------------------------------------------------------
# Decomposition (Component 2)
# --------------------------------------------------------------------------

@dataclass
class SubQuery:
    text: str
    topic_hint: Optional[str] = None  # cheap keyword-taxonomy label, if any


# --------------------------------------------------------------------------
# Corpus / retrieval (Component 3)
# --------------------------------------------------------------------------

@dataclass
class CorpusChunk:
    doc_id: str
    section_id: str
    section_title: str
    text: str
    chunk_index: int = 0

    @property
    def citation(self) -> str:
        return f"{self.doc_id} §{self.section_id}"


@dataclass
class ScoredChunk:
    chunk: CorpusChunk
    score: float
    sub_query: str


@dataclass
class RetrievalEvent:
    timestamp_s: float
    query: str
    trigger: str  # "provisional" | "multi_intent" | "refinement"
    result_count: int = 0
    event_id: str = field(default_factory=lambda: new_id("ret"))


# --------------------------------------------------------------------------
# Synthesis / session state (Component 4)
# --------------------------------------------------------------------------

@dataclass
class Claim:
    """One grounded factual assertion, traceable to a sub-query and citations."""

    claim_id: str
    sub_query: str
    text: str
    citations: list[str]
    supported: bool
    version: int = 1
    # Per-citation extractive evidence: [{"citation": "Doc_03 §2", "text": "<sentence lifted from that chunk>"}, ...]
    # Populated by SynthesisEngine._build_claim; lets the UI show exactly which
    # quoted corpus sentence backs which citation, instead of only the fused
    # claim text. Optional/empty for anything built before this field existed.
    evidence: list[dict] = field(default_factory=list)


@dataclass
class AnswerVersion:
    version: int
    text: str
    citations: list[str]
    uncertainty: Optional[str]
    created_at_s: float = field(default_factory=time.time)
    changed_claim_ids: list[str] = field(default_factory=list)
    # Set only when synthesize_refinement had to fall back to "most
    # recently active claim" with 2+ open topics and no topic/citation
    # signal telling it which one a late constraint actually modifies (see
    # benchmark/report.md edge case #2). Kept separate from `uncertainty`
    # (which is about corpus COVERAGE) because this is about SESSION
    # ambiguity — the corpus fully supports the answer given, the system
    # just isn't certain the answer was applied to the topic the user
    # meant, and says so instead of silently guessing.
    refinement_ambiguity: Optional[str] = None


@dataclass
class TelemetryEvent:
    event_type: str
    timestamp_s: float
    payload: dict
    event_id: str = field(default_factory=lambda: new_id("evt"))
