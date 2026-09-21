"""CopilotBriefEngine — orchestrates Components 1-5 into the end-to-end
streaming pipeline described in the Theme 4 guide's architecture diagram:

    Incoming Stream -> [1] Retrieval Controller -> [2] Multi-Intent Decomposer
        -> [3] Corpus Retrieval & Fusion -> [4] Session-Aware Synthesis
        -> Output: Streamed Answer + Grounded Citations + Observability Telemetry

One CopilotBriefEngine instance holds the shared corpus index, controller,
decomposer, synthesis engine, session store, and telemetry logger for the
lifetime of the process; `process_chunk` is the single entry point every
caller (CLI simulator, FastAPI WebSocket handler, tests) drives.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

from .controller import RetrievalController
from .corpus import load_corpus, load_corpus_catalog
from .decomposer import MultiIntentDecomposer
from .models import AnswerVersion, ControllerDecision, TranscriptChunk
from .retrieval import HybridRetriever
from .session import SessionState, SessionStore
from .synthesis import SynthesisEngine
from .telemetry import TelemetryLogger

_REFINEMENT_CUE_RE = re.compile(
    r"\b(actually|correction|one more thing|wait,|by the way|it was also|"
    r"turns out|update:|also,\s|forgot to mention)\b",
    re.IGNORECASE,
)


def _looks_like_refinement(text: str, has_prior_answer: bool) -> bool:
    """Heuristic: is this utterance a late-arriving constraint on the
    CURRENT topic thread, rather than a brand-new unrelated question?

    Two independent signals, either is sufficient:
      1. An explicit amendment cue phrase ("actually...", "correction...").
      2. A short, non-interrogative follow-up (no "?" / WH-question form,
         under ~12 tokens) — real amendments tend to be brief statements,
         not fresh questions.
    """

    if not has_prior_answer:
        return False
    if _REFINEMENT_CUE_RE.search(text):
        return True
    is_question = "?" in text or bool(re.match(r"^\s*(what|how|when|where|why|who|which|do|does|is|are|can|could|should)\b", text, re.IGNORECASE))
    short_statement = len(text.split()) <= 12
    return short_statement and not is_question


@dataclass
class PipelineStepResult:
    controller_decision: ControllerDecision
    controller_reason: str
    answer: Optional[AnswerVersion]
    is_utterance_end: bool
    # Which Component-4 path produced `answer` this step, for the UI to label
    # accurately (e.g. the "Live Refinement" diff view should only render on
    # "refinement", not on every multi-intent answer): one of
    # "multi_intent" | "provisional" | "refinement" | "presentation" |
    # "utterance_end_fallback" | None (no answer was produced this step).
    synthesis_mode: Optional[str] = None


class CopilotBriefEngine:
    def __init__(self, corpus_dir=None):
        chunks = load_corpus(corpus_dir)
        self.retriever = HybridRetriever(chunks)
        self.controller = RetrievalController()
        self.decomposer = MultiIntentDecomposer()
        self.telemetry = TelemetryLogger()
        self.synthesis = SynthesisEngine(self.retriever, self.telemetry)
        self.sessions = SessionStore()
        self._utterance_open: dict[str, bool] = {}
        self._utterance_synthesized: dict[str, bool] = {}
        # Doc/section title catalog for the UI (Evidence panel, citation
        # tooltips) — parsed independently of the chunking pass in
        # load_corpus, see corpus.load_corpus_catalog's docstring.
        self.corpus_catalog = load_corpus_catalog(corpus_dir)

    # ------------------------------------------------------------------

    def set_telemetry_sink(self, sink) -> None:
        self.telemetry.set_sink(sink)

    def reset_session(self, session_id: str) -> None:
        self.sessions.clear(session_id)
        self.controller.reset_utterance(session_id)
        self._utterance_open[session_id] = False

    # ------------------------------------------------------------------

    def process_chunk(self, chunk: TranscriptChunk) -> PipelineStepResult:
        session = self.sessions.get_or_create(chunk.session_id)

        if not self._utterance_open.get(chunk.session_id, False):
            self.controller.reset_utterance(chunk.session_id)
            self._utterance_open[chunk.session_id] = True
            self._utterance_synthesized[chunk.session_id] = False

        verdict = self.controller.evaluate(chunk, has_prior_answer=session.has_prior_answer)
        self.telemetry.log(session, "controller_decision", {
            "decision": verdict.decision.value,
            "reason": verdict.reason,
            "accumulated_text": verdict.accumulated_text,
        }, timestamp_s=chunk.timestamp_s)

        answer: Optional[AnswerVersion] = None
        synthesis_mode: Optional[str] = None

        if verdict.decision == ControllerDecision.NO_RETRIEVAL:
            answer = self.synthesis.synthesize_presentation_only(session, chunk.text, chunk.timestamp_s)
            synthesis_mode = "presentation"

        elif verdict.decision in (ControllerDecision.PROVISIONAL_RETRIEVE, ControllerDecision.MULTI_INTENT_RETRIEVE):
            sub_queries = self.decomposer.decompose(verdict.accumulated_text)
            trigger = "provisional" if verdict.decision == ControllerDecision.PROVISIONAL_RETRIEVE else "multi_intent"

            # Refinement-vs-fresh is only evaluated ONCE per utterance turn
            # (on whichever chunk first triggers synthesis). A later
            # multi-intent decompose within the SAME still-open utterance
            # is a continuation of this turn's answer, not a refinement of
            # the PREVIOUS turn's — so it always goes through synthesize_fresh,
            # which fully replaces the active claim set for this turn.
            already_synthesized_this_turn = self._utterance_synthesized.get(chunk.session_id, False)
            if not already_synthesized_this_turn and _looks_like_refinement(chunk.text, session.has_prior_answer):
                answer = self.synthesis.synthesize_refinement(session, sub_queries, chunk.timestamp_s)
                synthesis_mode = "refinement"
            else:
                answer = self.synthesis.synthesize_fresh(session, sub_queries, trigger, chunk.timestamp_s)
                synthesis_mode = trigger
            self._utterance_synthesized[chunk.session_id] = True

        elif verdict.decision == ControllerDecision.WAIT and chunk.is_utterance_end:
            # Utterance ended before any stable entity ever triggered
            # retrieval (e.g. a very short or unusual phrasing). Don't leave
            # the user with nothing: force a best-effort fresh synthesis on
            # whatever text accumulated.
            sub_queries = self.decomposer.decompose(verdict.accumulated_text)
            already_synthesized_this_turn = self._utterance_synthesized.get(chunk.session_id, False)
            if not already_synthesized_this_turn and _looks_like_refinement(chunk.text, session.has_prior_answer):
                answer = self.synthesis.synthesize_refinement(session, sub_queries, chunk.timestamp_s)
                synthesis_mode = "refinement"
            else:
                answer = self.synthesis.synthesize_fresh(session, sub_queries, "utterance_end_fallback", chunk.timestamp_s)
                synthesis_mode = "utterance_end_fallback"
            self._utterance_synthesized[chunk.session_id] = True

        if chunk.is_utterance_end:
            self.telemetry.log(session, "utterance_end", {
                "accumulated_text": verdict.accumulated_text,
            }, timestamp_s=chunk.timestamp_s)
            self._utterance_open[chunk.session_id] = False

        return PipelineStepResult(
            controller_decision=verdict.decision,
            controller_reason=verdict.reason,
            answer=answer,
            is_utterance_end=chunk.is_utterance_end,
            synthesis_mode=synthesis_mode,
        )

    # ------------------------------------------------------------------

    def process_utterance(self, session_id: str, chunks: list[tuple[float, str]], final_end: bool = True) -> list[PipelineStepResult]:
        """Convenience helper for non-streaming callers (tests, CLI): feed a
        list of (timestamp_s, text) fragments for one utterance in order."""

        results = []
        for i, (ts, text) in enumerate(chunks):
            is_end = final_end and (i == len(chunks) - 1)
            chunk = TranscriptChunk(session_id=session_id, text=text, timestamp_s=ts, is_utterance_end=is_end)
            results.append(self.process_chunk(chunk))
        return results

    def get_session(self, session_id: str) -> SessionState:
        return self.sessions.get_or_create(session_id)
