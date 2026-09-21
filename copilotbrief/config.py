"""Central configuration and tunable thresholds.

Nothing in this file is a query, a prompt, or a canned response — per the
Hard Engineering Rule "No Hardcoding / No Precomputation" in the Theme 4
guide, those must never live in application code. This file only holds
structural knobs (chunk sizes, weights, timing thresholds).
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = Path(os.environ.get("COPILOTBRIEF_CORPUS_DIR", PROJECT_ROOT / "corpus"))

# --- Retrieval controller (Component 1) ---------------------------------
# Minimum number of "stable" tokens (see controller.py) before a
# provisional retrieval is allowed to fire at all.
MIN_STABLE_TOKENS_FOR_PROVISIONAL = 5

# A chunk is considered semantically "unstable" (and the controller should
# WAIT) if it ends mid-clause on one of these dangling connective words.
DANGLING_CONNECTIVES = {
    "and", "or", "the", "a", "an", "to", "for", "of", "in", "on", "with",
    "is", "are", "was", "were", "i", "we", "need", "if", "that", "but",
}

# --- Multi-intent decomposer (Component 2) -------------------------------
# Coordinating / listing conjunctions used as compound-clause split points.
SPLIT_CONJUNCTIONS = {"and", "also", "plus", "as well as"}
MAX_SUBQUERIES = 5
MIN_SUBQUERY_TOKENS = 2

# --- Hybrid retrieval (Component 3) --------------------------------------
CHUNK_MAX_WORDS = 120
CHUNK_OVERLAP_WORDS = 20
TOP_K_PER_SUBQUERY = 8
RRF_K = 60  # reciprocal rank fusion constant
DENSE_MODEL_NAME = os.environ.get("COPILOTBRIEF_DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# The dense encoder is only ever loaded from the local HF cache at runtime
# (near-instant, no network). It is populated once, at Docker build time,
# by a RUN step that sets this flag to actually hit the network. This keeps
# every runtime start fast and deterministic (Gate G1) whether or not the
# grading machine has internet access, while still giving a genuine
# sentence-embedding "dense" signal whenever the image was built with one.
ALLOW_DENSE_MODEL_DOWNLOAD = os.environ.get("COPILOTBRIEF_ALLOW_MODEL_DOWNLOAD", "0") == "1"
MAX_CITATIONS_PER_ANSWER = 6

# --- Session-aware synthesis (Component 4) --------------------------------
GROUNDING_MIN_OVERLAP = 0.15  # minimum Dice-coefficient score (see synthesis._dice_coefficient) to accept a sentence as grounding support
MIN_SENTENCE_CONTENT_TOKENS = 6  # ignore very short sentences as grounding candidates: a 4-5 word sentence
# can score deceptively high on Dice purely by having a small denominator, without saying much
UNCERTAINTY_LABEL = "insufficient corpus evidence"

# --- Synthesis backend -----------------------------------------------------
# Extractive template synthesis is the default and needs no network access
# or API key, which keeps the system reproducible on a clean machine (Gate
# G1). Setting OPENAI_API_KEY enables an optional LLM-polish pass on top of
# the same grounded extractive content; the citations/claims are identical
# either way.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
USE_LLM_SYNTHESIS = bool(OPENAI_API_KEY) and os.environ.get("COPILOTBRIEF_DISABLE_LLM", "") != "1"
OPENAI_MODEL = os.environ.get("COPILOTBRIEF_OPENAI_MODEL", "gpt-4o-mini")

# --- Streaming simulation --------------------------------------------------
DEFAULT_PLAYBACK_SPEED = float(os.environ.get("COPILOTBRIEF_PLAYBACK_SPEED", "0"))  # 0 = no sleep, replay instantly
