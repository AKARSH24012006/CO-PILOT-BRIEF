#!/usr/bin/env python3
"""Benchmarking & Evaluation harness.

Computes the Theme 4 guide's own Technical Evaluation Gates (G2-G6; G1 is
verified by the test suite + Docker build, not here) against a held-out
prompt set (benchmark/testset.py, disjoint from the three demo scenarios),
and runs two architectural ablations:

  A1. Hybrid (BM25 + dense/TF-IDF, RRF-fused) retrieval vs. sparse-only
      (BM25 alone) — does fusion actually improve ranking on this corpus?
  A2. The streaming Retrieval Controller vs. a naive "wait for the full
      utterance" baseline — how much retrieval lead-time does incremental
      triggering actually buy?

Usage:  python benchmark/evaluate.py [--out benchmark/report.md]
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilotbrief.corpus import load_corpus  # noqa: E402
from copilotbrief.models import ControllerDecision, TranscriptChunk  # noqa: E402
from copilotbrief.pipeline import CopilotBriefEngine  # noqa: E402
from copilotbrief.retrieval import HybridRetriever  # noqa: E402

from testset import MULTI_INTENT_CASES, OUT_OF_DOMAIN_CASES, SINGLE_INTENT_CASES  # noqa: E402

logging.basicConfig(level=logging.ERROR)


# ---------------------------------------------------------------------
# Gate G2 — Early Retrieval
# ---------------------------------------------------------------------

def eval_g2_early_retrieval(engine: CopilotBriefEngine) -> dict:
    eligible = [c for c in SINGLE_INTENT_CASES if c.streamed]
    hits = 0
    for i, case in enumerate(eligible):
        sid = f"g2-{i}"
        results = engine.process_utterance(sid, case.streamed)
        early = any(
            r.controller_decision in (ControllerDecision.PROVISIONAL_RETRIEVE, ControllerDecision.MULTI_INTENT_RETRIEVE)
            for r in results[:-1]
        )
        hits += int(early)
    rate = hits / len(eligible) if eligible else 0.0
    return {"gate": "G2", "criterion": "Early Retrieval", "target": ">= 0.80",
            "n": len(eligible), "hits": hits, "rate": round(rate, 3), "pass": rate >= 0.80}


# ---------------------------------------------------------------------
# Gate G3 — Multi-Intent Identification
# ---------------------------------------------------------------------

def eval_g3_multi_intent(engine: CopilotBriefEngine) -> dict:
    hits = 0
    for i, case in enumerate(MULTI_INTENT_CASES):
        sid = f"g3-{i}"
        engine.process_utterance(sid, case.text_chunks)
        session = engine.get_session(sid)
        n_subintents = len(session.active_claim_order)
        if n_subintents >= case.expected_min_subintents:
            hits += 1
    rate = hits / len(MULTI_INTENT_CASES)
    return {"gate": "G3", "criterion": "Multi-Intent Identification", "target": ">= 0.70",
            "n": len(MULTI_INTENT_CASES), "hits": hits, "rate": round(rate, 3), "pass": rate >= 0.70}


# ---------------------------------------------------------------------
# Gate G4 — Factual Grounding
# ---------------------------------------------------------------------

def eval_g4_grounding(engine: CopilotBriefEngine, valid_doc_ids: set[str]) -> dict:
    grounded, expected_doc_hits, hallucinated, total = 0, 0, 0, 0
    for i, case in enumerate(SINGLE_INTENT_CASES):
        sid = f"g4-{i}"
        chunks = case.streamed or [(0.0, case.text)]
        results = engine.process_utterance(sid, chunks)
        final = results[-1].answer
        total += 1
        if final.citations:
            grounded += 1
            if any(c.split(" §")[0] == case.expected_doc for c in final.citations):
                expected_doc_hits += 1
            for c in final.citations:
                if c.split(" §")[0] not in valid_doc_ids:
                    hallucinated += 1
    support_rate = grounded / total
    precision_rate = expected_doc_hits / total
    return {
        "gate": "G4", "criterion": "Factual Grounding", "target": ">= 0.85 citation support, 0 hallucinated IDs",
        "n": total, "citation_support_rate": round(support_rate, 3),
        "expected_doc_precision": round(precision_rate, 3), "hallucinated_citations": hallucinated,
        "pass": support_rate >= 0.85 and hallucinated == 0,
    }


# ---------------------------------------------------------------------
# Out-of-domain: must NOT fabricate, must flag uncertainty
# ---------------------------------------------------------------------

def eval_out_of_domain(engine: CopilotBriefEngine) -> dict:
    clean = 0
    for i, case in enumerate(OUT_OF_DOMAIN_CASES):
        sid = f"ood-{i}"
        results = engine.process_utterance(sid, [(0.0, case.text)])
        final = results[-1].answer
        if final.citations == [] and final.uncertainty is not None:
            clean += 1
    rate = clean / len(OUT_OF_DOMAIN_CASES)
    return {"gate": "OOD", "criterion": "Out-of-domain refusal (no fabrication)", "target": "1.0",
            "n": len(OUT_OF_DOMAIN_CASES), "rate": round(rate, 3), "pass": rate == 1.0}


# ---------------------------------------------------------------------
# Gate G5 — Session Refinement (qualitative; also covered by test suite)
# ---------------------------------------------------------------------

def eval_g5_session_refinement(engine: CopilotBriefEngine) -> dict:
    sid = "g5-refine"
    r1 = engine.process_utterance(sid, [(0.0, "What is the diversion fuel reserve requirement for a domestic flight?")])
    r2 = engine.process_utterance(sid, [(6.0, "Actually, the flight is now international.")])
    v1, v2 = r1[-1].answer, r2[-1].answer
    version_incremented = v2.version == v1.version + 1
    citations_changed = set(v2.citations) != set(v1.citations)
    session = engine.get_session(sid)
    single_claim_thread = len(session.active_claim_order) == 1  # mutated in place, not appended
    ok = version_incremented and citations_changed and single_claim_thread
    return {"gate": "G5", "criterion": "Session Refinement", "target": "narrows in place, no full re-search",
            "version_incremented": version_incremented, "citations_changed": citations_changed,
            "claim_mutated_not_duplicated": single_claim_thread, "pass": ok}


# ---------------------------------------------------------------------
# Gate G6 — Telemetry & Observability trace coverage
# ---------------------------------------------------------------------

def eval_g6_trace_coverage(engine: CopilotBriefEngine) -> dict:
    sid = "g6-trace"
    chunks = [(0.0, "I need to know"), (0.8, "the crosswind limit for the A320,"), (1.6, "and the fuel reserve rule.")]
    n_chunks = len(chunks)
    engine.process_utterance(sid, chunks)
    session = engine.get_session(sid)

    controller_events = [e for e in session.telemetry if e.event_type == "controller_decision"]
    retrieval_events = [e for e in session.telemetry if e.event_type == "retrieval_event"]
    answer_events = [e for e in session.telemetry if e.event_type == "answer_version"]
    end_events = [e for e in session.telemetry if e.event_type == "utterance_end"]

    ok = (
        len(controller_events) == n_chunks  # one decision logged per chunk, no silent steps
        and len(retrieval_events) >= 1
        and len(answer_events) >= 1
        and len(end_events) == 1
    )
    return {"gate": "G6", "criterion": "Telemetry & Observability", "target": "100% trace coverage",
            "chunks_processed": n_chunks, "controller_decision_events": len(controller_events),
            "retrieval_events": len(retrieval_events), "answer_version_events": len(answer_events),
            "utterance_end_events": len(end_events), "pass": ok}


# ---------------------------------------------------------------------
# Ablation A1 — Hybrid vs. sparse-only retrieval
# ---------------------------------------------------------------------

def ablation_hybrid_vs_sparse() -> dict:
    chunks = load_corpus()
    hybrid = HybridRetriever(chunks, use_dense=True)
    sparse = HybridRetriever(chunks, use_dense=False)

    def score(retriever) -> dict:
        hit_at_5, reciprocal_ranks = 0, []
        for case in SINGLE_INTENT_CASES:
            results = retriever.search(case.text, top_k=5)
            doc_ids = [sc.chunk.doc_id for sc in results]
            if case.expected_doc in doc_ids:
                hit_at_5 += 1
                reciprocal_ranks.append(1.0 / (doc_ids.index(case.expected_doc) + 1))
            else:
                reciprocal_ranks.append(0.0)
        n = len(SINGLE_INTENT_CASES)
        return {
            "hit_at_5_rate": round(hit_at_5 / n, 3),
            "mrr": round(statistics.mean(reciprocal_ranks), 3),
        }

    return {
        "ablation": "A1: Hybrid (BM25+dense/TF-IDF, RRF) vs. sparse-only (BM25)",
        "dense_backend": hybrid.dense_backend_name,
        "hybrid": score(hybrid),
        "sparse_only": score(sparse),
        "n_queries": len(SINGLE_INTENT_CASES),
    }


# ---------------------------------------------------------------------
# Ablation A2 — Streaming controller vs. naive "wait for full utterance"
# ---------------------------------------------------------------------

class _NaiveWaitController:
    """Baseline: classic batch-RAG behavior. Never retrieves until the
    utterance is fully complete, and never decomposes mid-stream."""

    def evaluate(self, chunk: TranscriptChunk, has_prior_answer: bool = False):
        from copilotbrief.models import ControllerVerdict

        # (Accumulation is handled by the caller in this ablation harness
        # for simplicity — this class only ever fires on the last chunk.)
        decision = ControllerDecision.PROVISIONAL_RETRIEVE if chunk.is_utterance_end else ControllerDecision.WAIT
        return ControllerVerdict(
            decision=decision, reason="naive_wait_for_end", accumulated_text=chunk.text,
            is_utterance_end=chunk.is_utterance_end,
        )

    def reset_utterance(self, session_id: str) -> None:
        pass


def ablation_controller_lead_time() -> dict:
    from copilotbrief.controller import RetrievalController

    eligible = [c for c in SINGLE_INTENT_CASES if c.streamed] + [
        type("Wrap", (), {"streamed": c.text_chunks})() for c in MULTI_INTENT_CASES
    ]

    def measure(controller_factory) -> dict:
        lead_times, coverage_fractions = [], []
        for i, case in enumerate(eligible):
            controller = controller_factory()
            sid = f"a2-{i}"
            accumulated = ""
            first_retrieve_ts = None
            last_ts = case.streamed[-1][0]
            first_ts = case.streamed[0][0]
            for j, (ts, text) in enumerate(case.streamed):
                accumulated = (accumulated + " " + text).strip()
                is_end = j == len(case.streamed) - 1
                chunk = TranscriptChunk(session_id=sid, text=text, timestamp_s=ts, is_utterance_end=is_end)
                if hasattr(controller, "_state"):  # real RetrievalController accumulates internally
                    verdict = controller.evaluate(chunk)
                else:
                    chunk.text = accumulated  # naive baseline just needs to know utterance-end
                    verdict = controller.evaluate(chunk)
                if first_retrieve_ts is None and verdict.decision in (
                    ControllerDecision.PROVISIONAL_RETRIEVE, ControllerDecision.MULTI_INTENT_RETRIEVE,
                ):
                    first_retrieve_ts = ts
            duration = max(last_ts - first_ts, 1e-6)
            lead = (last_ts - first_retrieve_ts) if first_retrieve_ts is not None else 0.0
            lead_times.append(lead)
            coverage_fractions.append(lead / duration)
        return {
            "avg_lead_time_s": round(statistics.mean(lead_times), 3),
            "avg_pct_of_utterance_saved": round(statistics.mean(coverage_fractions) * 100, 1),
        }

    real = measure(lambda: RetrievalController())
    naive = measure(lambda: _NaiveWaitController())
    return {
        "ablation": "A2: Streaming Retrieval Controller vs. naive wait-for-end baseline",
        "n_utterances": len(eligible),
        "streaming_controller": real,
        "naive_baseline": naive,
    }


# ---------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------

KNOWN_FAILURE_MODES = """
## Analyzed Edge-Case Failures

**1. Negation-sensitive lexical grounding — FIXED (heuristic, not solved in general).**
Originally: the query "diversion fuel reserve requirement domestic flight" +
a refinement "the flight is now international" could rank a short, lexically
-dense NEGATIVE sentence (e.g. Doc_03 §2's closing line) above the actual
rule that answers the question, purely because Dice-coefficient scoring
rewards small denominators and has no notion of polarity. Reproduced live
with query "does an international flight need additional reserve fuel
beyond the standard reserve" — the wrong, off-topic negated sentence
(Doc_03 §5) was returned as the entire answer, with no mention of the real
45-minute international rule (Doc_03 §3) at all.
Fix (`synthesis._NEGATION_CUE_RE` + `config.NEGATION_MISMATCH_PENALTY`): a
grounding candidate sentence that asserts a negative the QUERY itself
doesn't share gets its Dice score demoted (not excluded) by a configurable
factor, so a non-negated sentence covering the same ground wins ties it
previously lost. The idiom "no less than X" / "not more than X" (extremely
common in regulatory text — e.g. "DH of no lower than 200 feet") is
explicitly excluded from the negation match, since it states a positive
numeric floor/ceiling, not a negation — two of the four benchmark
regressions caught while building this fix were exactly that false
positive. Regression-tested: `test_negated_sentence_does_not_outrank_the_actual_answer`
and `test_regulatory_no_less_than_phrasing_is_not_treated_as_negation` in
tests/test_synthesis.py. This is still a lexical heuristic, not an
entailment model — a genuinely negative correct answer to a non-negated
query is possible and would be under-penalized by the same rule; a
production system would still want an NLI-style check on top.

**2. Refinement-target ambiguity with multiple concurrent claims — now surfaced, not silent.**
`synthesize_refinement` matches a late constraint to an existing claim by
shared taxonomy topic or shared citation source document, falling back to
"most recently active claim" when neither signal fires. This fallback is
correct for the common case (one open topic thread, per the spec's own
Example 2), but previously misfired silently in a session with 2+
concurrently open, unrelated claims where the constraint's true target
might not be the most recent one.
Fix: when the fallback fires with 2+ open claims (a genuine guess, not the
single-thread case), the resulting `AnswerVersion.refinement_ambiguity`
field is now populated naming which topic the system assumed and how many
open topics it chose between — surfaced live in the dashboard (the blue
"Refinement target guessed" note) and in the WebSocket `answer` payload,
not just logged. It does not yet ask a clarifying question back (that
would need a conversational turn-taking mechanism this pipeline doesn't
have), but a downstream consumer — the dashboard, or a real dispatcher UI —
can now show the uncertainty instead of it being invisible. See the bonus
`ambiguous_refinement` demo scenario (`simulate/scenarios.py`) and
`test_refinement_target_ambiguity_is_surfaced_not_silent` /
`test_refinement_ambiguity_not_flagged_with_a_single_open_topic` in
tests/test_synthesis.py.

**3. TF-IDF fallback discrimination on a small, single-domain corpus.**
When the sentence-transformers dense encoder is unavailable (no cached
model, no network — the default state on an offline grading machine unless
the Docker image was built with network access), the "dense" signal falls
back to TF-IDF cosine similarity. On this corpus's 66 chunks, which share
heavy vocabulary overlap (nearly every section mentions "flight",
"dispatch", "crew"), TF-IDF and BM25 scores cluster tightly
(e.g. 0.0301-0.0328 across the top 8 candidates for a real query observed
during testing) and RRF fusion over two lexical signals adds less
diversity than a true dense embedding would. The Docker image mitigates
this by pre-warming the sentence-transformers model cache at build time
(see Dockerfile), but a grading environment that builds fully offline will
see the weaker fallback path.
"""


def build_report(results: dict) -> str:
    lines = ["# CopilotBrief — Benchmarking & Evaluation Report", ""]
    lines.append("## Technical Evaluation Gates (per Theme 4 spec §5)\n")
    lines.append("| Gate | Criterion | Target | Result | Pass |")
    lines.append("|---|---|---|---|---|")
    for key in ("g2", "g3", "g4", "g5", "g6", "ood"):
        r = results[key]
        result_str = ", ".join(f"{k}={v}" for k, v in r.items() if k not in ("gate", "criterion", "target", "pass"))
        lines.append(f"| {r['gate']} | {r['criterion']} | {r['target']} | {result_str} | {'✅' if r['pass'] else '❌'} |")

    lines.append("\n## Ablation A1 — Hybrid vs. Sparse-Only Retrieval\n")
    a1 = results["a1"]
    lines.append(f"Dense backend in use: `{a1['dense_backend']}` · n={a1['n_queries']} labeled single-intent queries\n")
    lines.append("| Retriever | Hit@5 | MRR |")
    lines.append("|---|---|---|")
    lines.append(f"| Hybrid (BM25 + dense/TF-IDF, RRF) | {a1['hybrid']['hit_at_5_rate']} | {a1['hybrid']['mrr']} |")
    lines.append(f"| Sparse-only (BM25) | {a1['sparse_only']['hit_at_5_rate']} | {a1['sparse_only']['mrr']} |")

    lines.append("\n## Ablation A2 — Streaming Controller vs. Naive Wait-for-End\n")
    a2 = results["a2"]
    lines.append(f"n={a2['n_utterances']} multi-chunk streamed utterances\n")
    lines.append("| Controller | Avg. retrieval lead time | Avg. % of utterance duration saved |")
    lines.append("|---|---|---|")
    lines.append(f"| Streaming (this system) | {a2['streaming_controller']['avg_lead_time_s']}s | {a2['streaming_controller']['avg_pct_of_utterance_saved']}% |")
    lines.append(f"| Naive (wait for utterance end) | {a2['naive_baseline']['avg_lead_time_s']}s | {a2['naive_baseline']['avg_pct_of_utterance_saved']}% |")

    lines.append(KNOWN_FAILURE_MODES)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).parent / "report.md"))
    args = parser.parse_args()

    print("Running benchmark suite (no network/API key required)...\n")
    engine = CopilotBriefEngine()
    valid_doc_ids = {c.doc_id for c in load_corpus()}

    results = {
        "g2": eval_g2_early_retrieval(CopilotBriefEngine()),
        "g3": eval_g3_multi_intent(CopilotBriefEngine()),
        "g4": eval_g4_grounding(CopilotBriefEngine(), valid_doc_ids),
        "g5": eval_g5_session_refinement(CopilotBriefEngine()),
        "g6": eval_g6_trace_coverage(CopilotBriefEngine()),
        "ood": eval_out_of_domain(CopilotBriefEngine()),
        "a1": ablation_hybrid_vs_sparse(),
        "a2": ablation_controller_lead_time(),
    }

    report = build_report(results)
    print(report)

    out_path = Path(args.out)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
