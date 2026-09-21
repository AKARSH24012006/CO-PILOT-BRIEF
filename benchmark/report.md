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

**1. Negation-sensitive lexical grounding (mitigated, not solved).**
Early in development, the query "diversion fuel reserve requirement domestic
flight" + a refinement "the flight is now international" ranked the sentence
*"No additional international reserve is required."* (Doc_03 §2 — the
domestic rule's closing sentence) above the actual international rule
(Doc_03 §3, the 45-minute reserve), because it lexically matches
"international" and "reserve" densely in a very short sentence. Dice-coefficient
scoring plus a minimum sentence-length floor (`MIN_SENTENCE_CONTENT_TOKENS`)
fixed this specific case, but the extractive grounder has no semantic
understanding of negation/polarity in general — a corpus sentence that
lexically overlaps a query while asserting the *opposite* of what's true can
still outrank the correct one. A production system would want an NLI-style
entailment check on top of retrieval, not just lexical overlap.

**2. Refinement-target ambiguity with multiple concurrent claims.**
`synthesize_refinement` matches a late constraint to an existing claim by
shared taxonomy topic or shared citation source document, falling back to
"most recently active claim" when neither signal fires (a deliberately
under-specified amendment like "the flight is now international" has no
retrievable evidence on its own to compare against). This fallback is
correct for the common case (one open topic thread, per the spec's own
Example 2) but would misfire in a session with two or more concurrently
open, unrelated claims where the constraint's true target is NOT the most
recent one — e.g. "...and also, use the B737 numbers for the crosswind
question" arriving after three other topics have since been discussed. The
system has no clarification-request path for this ambiguity; it silently
picks a target rather than asking.

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
