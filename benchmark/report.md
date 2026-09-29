# CopilotBrief — Benchmarking & Evaluation Report

## Technical Evaluation Gates (per Theme 4 spec §5)

| Gate | Criterion | Target | Result | Pass |
|---|---|---|---|---|
| G2 | Early Retrieval | >= 0.80 | n=3, hits=3, rate=1.0 | ✅ |
| G3 | Multi-Intent Identification | >= 0.70 | n=5, hits=5, rate=1.0 | ✅ |
| G4 | Factual Grounding | >= 0.85 citation support, 0 hallucinated IDs | n=18, citation_support_rate=0.944, expected_doc_precision=0.944, hallucinated_citations=0 | ✅ |
| G5 | Session Refinement | narrows in place, no full re-search | version_incremented=True, citations_changed=True, claim_mutated_not_duplicated=True | ✅ |
| G6 | Telemetry & Observability | 100% trace coverage | chunks_processed=3, controller_decision_events=3, retrieval_events=3, answer_version_events=2, utterance_end_events=1 | ✅ |
| OOD | Out-of-domain refusal (no fabrication) | 1.0 | n=3, rate=1.0 | ✅ |

## Ablation A1 — Hybrid vs. Sparse-Only Retrieval

Dense backend in use: `tfidf` · n=18 labeled single-intent queries

| Retriever | Hit@5 | MRR |
|---|---|---|
| Hybrid (BM25 + dense/TF-IDF, RRF) | 1.0 | 0.944 |
| Sparse-only (BM25) | 1.0 | 0.972 |

## Ablation A2 — Streaming Controller vs. Naive Wait-for-End

n=8 multi-chunk streamed utterances

| Controller | Avg. retrieval lead time | Avg. % of utterance duration saved |
|---|---|---|
| Streaming (this system) | 1.312s | 100.0% |
| Naive (wait for utterance end) | 0.0s | 0.0% |

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
