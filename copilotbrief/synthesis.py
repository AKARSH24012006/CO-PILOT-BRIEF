"""Component 4 — Session-Aware Synthesis & Refinement.

Builds a grounded, cited answer from retrieved corpus chunks, and — this is
the piece most naive RAG systems skip — refines an EXISTING answer in
place when a late-arriving constraint modifies part of an already-answered
question, instead of discarding session context and starting over
(Example 2 in the Theme 4 guide, and the "Context Loss on Late Constraints"
pitfall it warns about).

Grounding strategy (extractive-first, LLM-polish optional):
Every claim's TEXT is built by lifting the highest-overlap sentence(s) out
of the actual retrieved corpus chunk(s), not generated freestanding and
then merely "checked" against the corpus after the fact. That keeps
factual grounding structural rather than aspirational: a claim's wording
traces directly back into cited text. An optional LLM pass (only enabled
when OPENAI_API_KEY is set) may re-word the combined claims into smoother
prose, but is instructed to add no new facts and is skipped entirely by
default — the system is fully reproducible with zero external calls
(Gate G1).
"""

from __future__ import annotations

import re
import time
import uuid

from . import config
from .models import AnswerVersion, Claim, RetrievalEvent, SubQuery
from .retrieval import HybridRetriever, tokenize
from .session import SessionState
from .telemetry import TelemetryLogger

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# See config.NEGATION_MISMATCH_PENALTY for the reasoning. Matches whole-word
# negation cues only ("not", not "notable"); "n't" is handled separately
# since tokenization/regex word boundaries don't treat an apostrophe as a
# letter. "no"/"not" are excluded when followed by a comparative + "than"
# ("no less than 550 meters", "not more than 10 minutes") — that idiom
# states a positive numeric floor/ceiling, not a negation of the query's
# topic, and is extremely common in this kind of regulatory corpus text
# (two of the four benchmark regressions caught in review were exactly
# this false-positive pattern).
_NEGATION_CUE_RE = re.compile(
    r"\b(?:no|not)\b(?!\s+(?:less|lower|fewer|more|greater|earlier|later)\s+than)"
    r"|\b(?:never|without|cannot|neither|nor)\b"
    r"|n't",
    re.IGNORECASE,
)

# A late-arriving constraint often doesn't just ADD information, it
# SUPERSEDES a qualifier in the original question ("domestic" -> now
# "international"). Carrying the stale qualifier into the delta-retrieval
# query dilutes/misdirects it toward the now-superseded section instead of
# the one that actually applies. This is a small, domain-agnostic-style
# antonym list, not a canned answer — it only reshapes the *query text*,
# never the retrieved or returned content.
_SUPERSESSION_PAIRS = [
    ("domestic", "international"),
    ("dry", "wet"),
    ("dry", "contaminated"),
]


def _strip_superseded_qualifiers(original_text: str, constraint_text: str) -> str:
    lower_constraint = constraint_text.lower()
    result = original_text
    for a, b in _SUPERSESSION_PAIRS:
        if b in lower_constraint and re.search(rf"\b{a}\b", result, re.IGNORECASE):
            result = re.sub(rf"\b{a}\b", "", result, flags=re.IGNORECASE)
        if a in lower_constraint and re.search(rf"\b{b}\b", result, re.IGNORECASE):
            result = re.sub(rf"\b{b}\b", "", result, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", result).strip()
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "for", "in",
    "on", "and", "or", "that", "this", "it", "be", "by", "at", "as", "with",
}


def _content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOP and len(t) > 1}


def _split_sentences(text: str) -> list[str]:
    text = text.replace("\n", " ")
    parts = [p.strip().lstrip("-•").strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    parts = [p for p in parts if p]
    return parts or [text.strip()]


def _dice_coefficient(query_tokens: set[str], sentence_tokens: set[str]) -> float:
    """Sorensen-Dice overlap: 2*|A∩B| / (|A|+|B|).

    Chosen over plain Jaccard (|A∩B|/|A∪B|), which over-punishes long,
    genuinely relevant corpus sentences purely for having a larger
    vocabulary, and over plain recall/overlap-coefficient
    (|A∩B|/min(|A|,|B|)), which over-*rewards* long sentences that
    happen to reuse a handful of generic query words (e.g. "flight",
    "domestic") without being specifically on-topic. Dice sits between
    the two and tracks intuitive relevance much better on this corpus'
    section-length sentences."""

    if not query_tokens or not sentence_tokens:
        return 0.0
    inter = len(query_tokens & sentence_tokens)
    denom = len(query_tokens) + len(sentence_tokens)
    return (2 * inter) / denom if denom else 0.0


class SynthesisEngine:
    def __init__(self, retriever: HybridRetriever, telemetry: TelemetryLogger):
        self.retriever = retriever
        self.telemetry = telemetry

    # -- grounded claim construction -----------------------------------

    def _build_claim(self, sub_query: SubQuery, claim_id: str, version: int) -> tuple[Claim, RetrievalEvent]:
        scored = self.retriever.search(sub_query.text, top_k=config.TOP_K_PER_SUBQUERY)
        query_tokens = _content_tokens(sub_query.text)
        query_is_negated = bool(_NEGATION_CUE_RE.search(sub_query.text))

        best_sentences: list[tuple[float, str, str]] = []  # (score, sentence, citation)
        for sc in scored:  # inspect every retrieved candidate (already top_k-limited by the retriever)
            for sentence in _split_sentences(sc.chunk.text):
                sentence_tokens = _content_tokens(sentence)
                if len(sentence_tokens) < config.MIN_SENTENCE_CONTENT_TOKENS:
                    continue
                score = _dice_coefficient(query_tokens, sentence_tokens)
                if score > 0 and not query_is_negated and _NEGATION_CUE_RE.search(sentence):
                    # Sentence asserts a negative the query didn't ask for —
                    # demote rather than exclude (see config.NEGATION_MISMATCH_PENALTY).
                    score *= config.NEGATION_MISMATCH_PENALTY
                if score > 0:
                    best_sentences.append((score, sentence, sc.chunk.citation))

        best_sentences.sort(key=lambda x: -x[0])
        top = [s for s in best_sentences if s[0] >= config.GROUNDING_MIN_OVERLAP][:2]

        if top:
            claim_text = " ".join(s[1] for s in top)
            citations = list(dict.fromkeys(s[2] for s in top))  # dedupe, preserve order
            supported = True
            evidence = [{"citation": s[2], "text": s[1]} for s in top]
        else:
            claim_text = f"No corpus evidence sufficiently addresses: \"{sub_query.text}\"."
            citations = []
            supported = False
            evidence = []

        claim = Claim(
            claim_id=claim_id,
            sub_query=sub_query.text,
            text=claim_text,
            citations=citations,
            supported=supported,
            version=version,
            evidence=evidence,
        )
        retrieval_event = RetrievalEvent(
            timestamp_s=time.time(),
            query=sub_query.text,
            trigger="pending",  # caller overwrites with the real trigger
            result_count=len(scored),
        )
        return claim, retrieval_event

    # -- rendering --------------------------------------------------------

    def _render(self, session: SessionState) -> tuple[str, list[str], str | None]:
        supported_texts = []
        unsupported_topics = []
        citations: list[str] = []

        for claim_id in session.active_claim_order:
            claim = session.claims.get(claim_id)
            if claim is None:
                continue
            if claim.supported:
                supported_texts.append(claim.text)
                for c in claim.citations:
                    if c not in citations:
                        citations.append(c)
            else:
                unsupported_topics.append(claim.sub_query)

        answer_text = " ".join(supported_texts) if supported_texts else (
            "The retrieved corpus did not contain sufficient evidence to answer this request."
        )
        citations = citations[: config.MAX_CITATIONS_PER_ANSWER]

        uncertainty = None
        if unsupported_topics:
            uncertainty = "Could not be verified from the retrieved corpus: " + "; ".join(unsupported_topics) + "."

        return answer_text, citations, uncertainty

    # -- public entry points ------------------------------------------------

    def synthesize_fresh(
        self, session: SessionState, sub_queries: list[SubQuery], trigger: str, timestamp_s: float
    ) -> AnswerVersion:
        """A brand-new question (or the first / multi-intent resolution of
        this utterance). Replaces the active claim set entirely."""

        new_order: list[str] = []
        for sq in sub_queries:
            claim_id = f"claim_{uuid.uuid4().hex[:8]}"
            claim, retrieval_event = self._build_claim(sq, claim_id, version=1)
            retrieval_event.trigger = trigger
            retrieval_event.timestamp_s = timestamp_s
            session.claims[claim_id] = claim
            session.retrieval_events.append(retrieval_event)
            self.telemetry.log(session, "retrieval_event", {
                "query": sq.text, "trigger": trigger, "result_count": retrieval_event.result_count,
            }, timestamp_s=timestamp_s)
            new_order.append(claim_id)

        session.active_claim_order = new_order
        text, citations, uncertainty = self._render(session)
        version_num = session.current_version + 1
        answer = AnswerVersion(
            version=version_num, text=text, citations=citations, uncertainty=uncertainty,
            changed_claim_ids=list(new_order),
        )
        session.answer_versions.append(answer)
        self.telemetry.log(session, "answer_version", {
            "version": version_num, "changed_claim_ids": new_order, "uncertainty": uncertainty,
        }, timestamp_s=timestamp_s)
        return answer

    def synthesize_refinement(
        self, session: SessionState, constraint_sub_queries: list[SubQuery], timestamp_s: float
    ) -> AnswerVersion:
        """A late-arriving constraint on the CURRENT topic thread. Mutates
        only the claims whose topic the constraint touches (matched via
        retrieval, not via a lookup table); every other claim in
        session.active_claim_order is carried forward untouched."""

        from . import taxonomy  # local import: avoids a module-level cycle risk

        changed_ids: list[str] = []
        # (assumed target's sub_query, the constraint text applied to it) —
        # populated only when the fallback below had 2+ open topics to
        # choose between and no signal saying which one, so the guess is
        # genuinely ambiguous rather than the routine single-thread case.
        ambiguous_targets: list[tuple[str, str]] = []

        for csq in constraint_sub_queries:
            # Retrieve for the constraint ON ITS OWN first — this both tells
            # us what the constraint is actually about (via its citations'
            # source docs) and doubles as the claim content if it turns out
            # to be a genuinely new fact rather than a modification.
            probe_claim_id = f"claim_{uuid.uuid4().hex[:8]}"
            probe_claim, probe_event = self._build_claim(csq, probe_claim_id, version=1)
            probe_event.trigger = "refinement"
            probe_event.timestamp_s = timestamp_s
            session.retrieval_events.append(probe_event)
            self.telemetry.log(session, "retrieval_event", {
                "query": csq.text, "trigger": "refinement", "result_count": probe_event.result_count,
            }, timestamp_s=timestamp_s)

            constraint_topics = taxonomy.find_topics(csq.text)
            constraint_docs = {c.split(" §")[0] for c in probe_claim.citations}

            # Find an existing active claim this constraint modifies: same
            # taxonomy topic, OR its delta retrieval landed in the same
            # source document(s) as the existing claim (a stronger, more
            # robust signal than keyword-topic matching alone).
            target_claim_id = None
            for cid in session.active_claim_order:
                existing = session.claims.get(cid)
                if existing is None:
                    continue
                existing_topics = taxonomy.find_topics(existing.sub_query)
                existing_docs = {c.split(" §")[0] for c in existing.citations}
                if (constraint_topics & existing_topics) or (constraint_docs & existing_docs):
                    target_claim_id = cid
                    break

            if target_claim_id is None and session.active_claim_order:
                # No topic/doc signal (a short amendment like "the flight is
                # now international" is deliberately under-specified on its
                # own — it only makes sense merged with existing context).
                # The caller only routes here once it's already decided this
                # utterance IS a refinement of the ongoing thread, so the
                # most-recently-active claim is the reasonable default
                # target rather than treating it as an unrelated new fact.
                target_claim_id = session.active_claim_order[-1]
                if len(session.active_claim_order) > 1:
                    # More than one open topic AND no signal which one this
                    # constraint is about — the "most recent" guess is a
                    # real guess here, not just the obviously-correct single
                    # -thread case. Surface it instead of staying silent.
                    assumed = session.claims.get(target_claim_id)
                    if assumed is not None:
                        ambiguous_targets.append((assumed.sub_query, csq.text))

            if target_claim_id is not None:
                existing = session.claims[target_claim_id]
                # Re-run retrieval scoped to the ORIGINAL sub-question (minus
                # any qualifier the constraint just superseded) plus the new
                # constraint — a targeted delta query, not a full corpus
                # re-scan of every prior sub-query.
                base_text = _strip_superseded_qualifiers(existing.sub_query, csq.text)
                delta_query = SubQuery(text=f"{base_text} — {csq.text}", topic_hint=csq.topic_hint)
                new_claim, retrieval_event = self._build_claim(delta_query, target_claim_id, version=existing.version + 1)
                retrieval_event.trigger = "refinement"
                retrieval_event.timestamp_s = timestamp_s
                session.claims[target_claim_id] = new_claim
                session.retrieval_events.append(retrieval_event)
                self.telemetry.log(session, "retrieval_event", {
                    "query": delta_query.text, "trigger": "refinement", "result_count": retrieval_event.result_count,
                }, timestamp_s=timestamp_s)
                changed_ids.append(target_claim_id)
            else:
                # No existing claim covers this topic — it's a genuinely new
                # fact introduced by the constraint (e.g. a new mandatory
                # document requirement), so the probe claim itself is ADDED
                # to the thread rather than replacing anything.
                session.claims[probe_claim_id] = probe_claim
                session.active_claim_order.append(probe_claim_id)
                changed_ids.append(probe_claim_id)

        text, citations, uncertainty = self._render(session)

        refinement_ambiguity = None
        if ambiguous_targets:
            open_count = len(session.active_claim_order)
            notes = [
                f"applied \"{ctext}\" to \"{sq}\" (the most recently discussed of {open_count} open topics) "
                "— no topic or citation match tied it to a specific one; say which topic you meant if that's wrong."
                for sq, ctext in ambiguous_targets
            ]
            refinement_ambiguity = " ".join(notes)

        version_num = session.current_version + 1
        answer = AnswerVersion(
            version=version_num, text=text, citations=citations, uncertainty=uncertainty,
            changed_claim_ids=changed_ids, refinement_ambiguity=refinement_ambiguity,
        )
        session.answer_versions.append(answer)
        self.telemetry.log(session, "answer_version", {
            "version": version_num, "changed_claim_ids": changed_ids, "uncertainty": uncertainty,
            "refinement_ambiguity": refinement_ambiguity,
        }, timestamp_s=timestamp_s)
        return answer

    def synthesize_presentation_only(self, session: SessionState, instruction_text: str, timestamp_s: float) -> AnswerVersion:
        """No-Retrieval path (Example 3): reformat the CURRENT answer text
        without touching claims, citations, or issuing any corpus query."""

        prior = session.answer_versions[-1] if session.answer_versions else None
        base_text = prior.text if prior else ""
        citations = prior.citations if prior else []

        reformatted = self._reformat(base_text, instruction_text)

        version_num = session.current_version + 1
        answer = AnswerVersion(
            version=version_num, text=reformatted, citations=citations,
            uncertainty=prior.uncertainty if prior else None, changed_claim_ids=[],
        )
        session.answer_versions.append(answer)
        self.telemetry.log(session, "answer_version", {
            "version": version_num, "changed_claim_ids": [], "reason": "presentation_restructure",
        }, timestamp_s=timestamp_s)
        return answer

    @staticmethod
    def _reformat(text: str, instruction: str) -> str:
        """Pure presentation transform — no new facts, no corpus access."""
        sentences = [s for s in _split_sentences(text) if s]
        if re.search(r"bullet", instruction, re.IGNORECASE):
            return "\n".join(f"- {s}" for s in sentences)
        if re.search(r"shorter|concise|one\s+sentence", instruction, re.IGNORECASE):
            return sentences[0] if sentences else text
        return text
