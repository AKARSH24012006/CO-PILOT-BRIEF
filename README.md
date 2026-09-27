# CopilotBrief

**Streaming Live RAG for aviation pre-flight briefings.**
Built for the Samsung PRISM GenAI Hackathon 3.0 (3rd Edition) — **Theme 4: Streaming Live RAG**.

CopilotBrief is an event-driven retrieval engine for a dispatcher/pilot
pre-flight briefing assistant. It listens to a transcript as it streams in
(word by word, not turn by turn), starts searching the moment a question
becomes answerable instead of waiting for the speaker to finish, splits
compound questions ("what's the crosswind limit, do we need de-icing, and
what's the fuel reserve?") into parallel searches, and — the part most
demo RAG systems skip — revises an answer **in place** when the user adds
a late constraint ("actually, the flight is now international") instead of
throwing away context and starting over.

Every factual claim is grounded in an explicit citation back to a
synthetic airline operations manual (`corpus/`, 15 documents / 66
sections). When the corpus doesn't cover something, the system says so
instead of guessing — verified in `benchmark/evaluate.py`.

## Quickstart

No API key, no GPU, no internet access required at runtime.

```bash
# Option A — Docker (starts the live web demo on :8000)
docker compose up --build

# Option B — no Docker: CLI replay of the three canonical scenarios
pip install -r requirements.txt
python run_demo.py

# Option C — run the live demo server directly
uvicorn server.main:app --port 8000
# then open http://localhost:8000
```

```bash
# Tests (25 tests, ~4s, no network needed)
pytest tests/ -v

# Benchmark & evaluation gates (writes benchmark/report.md)
python benchmark/evaluate.py
```

An optional `OPENAI_API_KEY` environment variable enables an LLM prose-polish
pass on top of the same grounded, extractively-cited content (see
[Synthesis backend](#4-session-aware-synthesis--refinement) below) —
everything above works correctly with it unset.

## Demo video

[FILL IN: add your recorded demo video link here — YouTube or Drive, max 5
minutes, required by the submission checklist]

## Submission (Samsung PRISM GenAI Hackathon 3.0 — Theme 4)

- Theme: Theme 4, Streaming Live RAG
- Presentation: [`CopilotBrief_Submission.pptx`](CopilotBrief_Submission.pptx)
- AI usage disclosure: [`LangAI3.0_AI_Disclosure.docx`](LangAI3.0_AI_Disclosure.docx)
- Tag: `PRISM_GENAI_HACKATHON_Y2026`

## Why aviation dispatch

Theme 4's own scenario language — compound utterances, late-arriving
constraints, corpus-grounded answers with explicit uncertainty — maps
almost exactly onto a real pre-flight dispatch briefing: a pilot or
dispatcher rattles off several regulatory questions in one breath, and a
new fact ("this leg just became international") can change which rule
applies mid-conversation without invalidating everything already
established. It's also a domain where a *wrong but confident* answer has
real stakes, which makes the "explicit uncertainty over fabrication" rule
in the spec feel like the actual point of the system, not a compliance
checkbox.

The corpus (`corpus/*.md`) is a **synthetic operations manual for a
fictional carrier ("Meridian Air")** — invented section numbers and
numeric limits, not real FAA/DGCA/ICAO regulations. This is a deliberate
choice, not a shortcut: because nothing in the corpus matches any real
regulatory text word-for-word, any grounded answer is *provably* sourced
from the supplied corpus rather than the model's parametric memory, which
is exactly what Gate G4 (Factual Grounding) and the "Corpus Isolation"
rule are checking for.

## Architecture

```
Incoming Stream: [Chunk 0.0s] -> [Chunk 0.8s] -> [Chunk 1.6s] -> [Utterance End]
   |
   v
[1] Retrieval Controller      (controller.py)
    - Intent stability check over the growing transcript
    - Decision: Wait | Provisional-Retrieve | Multi-Intent-Retrieve | No-Retrieval
   |  (retrieve triggered)
   v
[2] Multi-Intent Decomposer   (decomposer.py, clausing.py)
    - Splits compound utterances into orthogonal sub-queries
    - Carries shared entities (aircraft type, location) into later clauses
   |
   v
[3] Corpus Retrieval & Fusion (retrieval.py)
    - BM25 (sparse) + sentence-embedding-or-TF-IDF (dense), RRF-fused
    - Deduplicated across sub-queries by (doc_id, section_id)
   |
   v
[4] Session-Aware Synthesis   (synthesis.py, session.py)
    - Extractive, sentence-level grounding with inline citations
    - Refines existing claims in place on late-arriving constraints
    - Explicit uncertainty when the corpus doesn't cover a sub-intent
   |
   v
Output: Streamed Answer + Grounded Citations + Observability Telemetry (telemetry.py)
```

`pipeline.py`'s `CopilotBriefEngine` wires all five components together and
is the single entry point every caller (CLI, WebSocket server, tests,
benchmark) drives through `process_chunk(chunk)`.

### 1. Retrieval Controller

Evaluates the accumulated transcript on every incoming fragment and picks
one of four decisions:

- **Wait** — the utterance is still semantically unstable (ends on a
  dangling connective like "in", "and", "to" with no concrete entity yet).
- **Provisional-Retrieve** — fires once a concrete, retrieval-worthy entity
  has stabilized (a domain keyword, a `N <unit>` quantity, a proper noun
  like an aircraft type), even if the speaker keeps talking.
- **Multi-Intent-Retrieve** — fires once the accumulated text contains
  list structure (an "and"/"also"/"plus" glue word) with 2+ distinct,
  content-bearing clauses.
- **No-Retrieval** — the turn is a pure presentation request ("repeat that
  in two bullets") on top of an existing answer; no corpus query is
  issued at all.

The two pitfalls the Theme 4 guide calls out by name are handled
explicitly rather than left to chance:

- *Eager/Premature Retrieval on Noise* — a lightweight gazetteer
  (`taxonomy.py`: domain keywords, `N <unit>` numeric patterns, proper
  nouns) gates Provisional-Retrieve on an actual entity being present, not
  just "enough tokens accumulated."
- *Ignoring Presentation-Only Turns* — a regex over reformatting verbs
  (repeat/shorten/bulletize/translate/...) combined with "does this
  session have a prior answer" routes straight to No-Retrieval, verified
  in `test_controller.py`.

### 2. Multi-Intent Decomposer

Splits a compound utterance into sub-queries using a **shared** clause
grammar (`clausing.py`) — shared with the controller specifically so the
controller's clause *count* and the decomposer's actual clause *split*
can never disagree with each other. Two rules mitigate the "Over-Fragmenting
Sub-Queries" pitfall:

- A bare comma is only treated as a list-item boundary once an
  "and"/"also"/"plus" is present elsewhere in the text — i.e. there's
  independent evidence this *is* a list, not just ordinary sentence
  punctuation ("For dispatch planning, what's the fuel rule?" stays one
  question).
- Every candidate clause needs a minimum content-token count after
  filler-stripping ("I need", "and I want", a leading "Before we release
  this flight,..." intro) — a dangling "I need" left over from a split
  never becomes its own spurious sub-query.

A second pass carries forward entities that appear only in the first
clause (an aircraft type, a location) into later clauses that don't repeat
them, so "...crosswind limit for the A320, and the fuel reserve rule" is
decomposed into two queries that are *both* still about the A320.

### 3. Corpus Retrieval & Fusion

Hybrid retrieval: BM25 (`rank-bm25`) for sparse lexical matching, fused via
**Reciprocal Rank Fusion** with a dense signal. The dense signal is
`sentence-transformers/all-MiniLM-L6-v2` when available, and **falls back
to TF-IDF cosine similarity** when it isn't (see
[Reproducibility](#reproducibility--offline-behavior) below) — the
retriever always performs a genuine two-signal fusion, it just degrades
the *source* of the second signal rather than hard-failing. `search_many()`
runs retrieval per sub-query and deduplicates candidates by
`(doc_id, section_id)`, keeping the highest score seen — the "Evidence
Fusion & Reranking" step.

### 4. Session-Aware Synthesis & Refinement

This is the component most naive streaming-RAG demos skip. Design choices:

- **Extractive-first grounding.** A claim's text is built by lifting the
  highest-scoring *sentence* out of the actual retrieved chunk(s) — not
  generated freestanding and checked against the corpus afterward. Scoring
  uses a **Sørensen–Dice coefficient** between query content-tokens and
  sentence content-tokens (`synthesis._dice_coefficient`), chosen after
  empirically observing that plain Jaccard over-punishes long-but-relevant
  sentences, while a pure overlap-coefficient over-rewards long sentences
  that merely reuse generic query vocabulary. A sentence below
  `GROUNDING_MIN_OVERLAP` is rejected rather than cited — the sub-question
  is marked with an explicit uncertainty note instead.
- **Claims are the unit of state**, not raw answer text. Each `Claim`
  (`models.py`) tracks its own sub-query, citations, support status, and
  version. An `AnswerVersion` is just a rendering of the session's current
  ordered claim list.
- **Refinement mutates claims in place.** `synthesize_refinement` retrieves
  for the constraint alone, matches it to an existing claim by shared
  taxonomy topic or shared citation source document (falling back to "the
  most recently active claim" when a short amendment like "the flight is
  now international" carries no standalone retrievable signal of its
  own — see `benchmark/report.md`'s edge-case #2 for this fallback's known
  limitation), then re-retrieves with a **targeted delta query**: the
  original sub-question with any now-superseded qualifier stripped
  (`_strip_superseded_qualifiers` — a tiny antonym list: domestic/
  international, dry/wet/contaminated) plus the new constraint. Every
  *other* claim in the session is left completely untouched — same
  `claim_id`, same text, same citations.
- **Optional LLM polish, never the source of truth.** When
  `OPENAI_API_KEY` is set, a rewrite pass may smooth the combined claim
  text into more natural prose; the citations and underlying claims are
  identical either way, and the pass is skipped entirely by default so the
  system needs zero external calls (Gate G1).

### 5. Observability Telemetry

Every controller decision, retrieval call, and answer version is logged
(`telemetry.py`) — 100% trace coverage by construction, since every branch
in `pipeline.process_chunk` logs through the same `TelemetryLogger` before
returning. `TelemetryLogger.structured_output_record(session)` assembles
the exact shape shown in the Theme 4 guide's own example:

```json
{
  "retrieval_events": [{"timestamp_s": 0.8, "query": "...", "trigger": "provisional", "result_count": 8}],
  "sub_queries": ["..."],
  "answer": "...",
  "citations": ["Doc_01 §3", "Doc_03 §2"],
  "uncertainty": null,
  "answer_version": 2,
  "answer_version_lineage": [{"version": 1, "changed_claims": [...], "created_at_s": ...}],
  "token_cost_estimate": 212
}
```

`token_cost_estimate` is a dependency-free ~0.75-words/token approximation
(no network call needed to report a cost estimate); real OpenAI usage is
substituted when the optional LLM-polish path runs.

## Reproducibility / offline behavior

Gate G1 asks for a container that launches with a single command on a
clean machine. The one part of this system that *could* need the network
(the sentence-embedding model) is handled so it never blocks startup:

1. The Docker build pre-warms the model into the image (`Dockerfile`);
   runtime then loads it from local cache in well under a second.
2. If the build itself has no network (fully air-gapped CI), the `RUN`
   step fails softly (`|| true`) and the image still builds.
3. At runtime, `retrieval.py` first tries a **local-cache-only** load
   (`local_files_only=True`, no network call at all). Only if that misses
   *and* `COPILOTBRIEF_ALLOW_MODEL_DOWNLOAD=1` is explicitly set (true only
   during the Docker build step) does it attempt a real download — so a
   cold, offline `python run_demo.py` never spends ~30s on retry backoff
   before giving up; it falls back to TF-IDF immediately.

This sandbox's own network egress is restricted (huggingface.co
unreachable), so every test run and demo shown while building this project
ran on the TF-IDF fallback path — which is exactly why that path got the
same testing attention as the primary one, not just an afterthought
exception handler. `docker build` was written defensively for this reason
but could not be executed end-to-end inside this development sandbox (no
Docker daemon available here); `python run_demo.py`, the full `pytest`
suite, and `benchmark/evaluate.py` were all run and verified directly.

## Evaluation results

`python benchmark/evaluate.py` scores the system against the Theme 4
spec's own Technical Evaluation Gates on a held-out prompt set (disjoint
from the three demo scenarios) and runs two architectural ablations. Full
output — including three analyzed edge-case failures — is written to
[`benchmark/report.md`](benchmark/report.md); at last run:

| Gate | Criterion | Target | Result |
|---|---|---|---|
| G2 | Early Retrieval | ≥ 0.80 | 1.0 (3/3) |
| G3 | Multi-Intent Identification | ≥ 0.70 | 1.0 (5/5) |
| G4 | Factual Grounding | ≥ 0.85 support, 0 hallucinated IDs | 0.944 support, 0 hallucinated |
| G5 | Session Refinement | narrows in place | ✅ |
| G6 | Telemetry Trace Coverage | 100% | ✅ |

**Ablation A1 (Hybrid vs. sparse-only retrieval):** on this corpus's 66
chunks, BM25-only actually edges out the RRF-fused hybrid on MRR (0.972
vs. 0.944) when the dense signal is running on its TF-IDF fallback — see
edge-case #3 in the report for why, and why this is expected to reverse
once a real sentence-embedding model is cached (the architecture, not the
corpus, is what Gate G2/G3 evaluate).

**Ablation A2 (streaming controller vs. naive wait-for-end):** the
streaming controller retrieves with an average lead time of ~1.3s before
the utterance finishes — on the held-out multi-chunk set, that's 100% of
the utterance's remaining duration reclaimed versus a batch-RAG baseline
that, by definition, always retrieves at 0% lead time.

## Repository layout

```
copilotbrief/          Core pipeline package
  controller.py           Component 1 — retrieval controller
  decomposer.py           Component 2 — multi-intent decomposer
  clausing.py              shared clause-splitting grammar (controller + decomposer)
  taxonomy.py               domain keyword/entity gazetteer (a signal, never an answer source)
  retrieval.py            Component 3 — hybrid BM25+dense/TF-IDF retrieval, RRF fusion
  synthesis.py             Component 4 — extractive grounding + refinement
  session.py                ephemeral, session-bound state store
  telemetry.py             Component 5 — structured event logging
  pipeline.py              CopilotBriefEngine — orchestrates 1-5
  corpus.py                 corpus loader / section-aware chunker
  models.py, config.py

corpus/                 15 synthetic ops-manual documents (66 sections)
simulate/               scenario fixtures + stream replay harness
server/                 FastAPI + WebSocket live demo, static/index.html frontend
tests/                  25 pytest tests (component + end-to-end)
benchmark/              held-out eval set, gates G2-G6, 2 ablations, report.md
run_demo.py             single-command CLI: replays the 3 canonical scenarios
Dockerfile, docker-compose.yml
```

## Known limitations

See `benchmark/report.md` → "Analyzed Edge-Case Failures" for three
concrete, reproduced failure modes (negation-sensitive lexical grounding,
refinement-target ambiguity across multiple concurrent claims, and TF-IDF
fallback discrimination on a small corpus) with the mitigations already in
place and what would still be needed for a production system.
