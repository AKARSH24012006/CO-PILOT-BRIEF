"""A small domain vocabulary used as a *signal*, never as an answer source.

This is a gazetteer of aviation-ops topic keywords. Two consumers use it:

  - controller.py: to detect that a *new, stable, on-topic entity* has just
    appeared in the streamed transcript (drives Wait -> Retrieve timing).
  - decomposer.py: to attach a cheap `topic_hint` label to each extracted
    sub-query, and to tell whether two clauses are about different topics
    (a signal for multi-intent splitting).

Nothing here answers a question or is returned to the user directly — every
factual claim in a response still has to come from a corpus chunk retrieved
at query time (see retrieval.py). This file only shapes *when* and *how* to
query, which keeps it outside the "No Hardcoding / No Precomputation" rule
(no prompts, no queries, no canned responses live here).
"""

from __future__ import annotations

import re

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "crosswind": ["crosswind", "cross wind", "gust", "wind limit", "wind component"],
    "deicing": ["de-icing", "deicing", "anti-icing", "antiicing", "holdover", "frost", "ice on the wing"],
    "fuel_reserve": ["fuel reserve", "diversion fuel", "final reserve", "contingency fuel", "trip fuel", "edto", "etops"],
    "weather_minima": ["rvr", "visibility minima", "decision height", "landing minima", "takeoff minima", "ceiling"],
    "holding": ["holding pattern", "hold at", "holding fuel", "holding speed"],
    "notam": ["notam", "runway closure", "taxiway closure", "navaid outage"],
    "international": ["international", "overflight", "customs", "immigration", "passport", "visa", "border"],
    "alternate": ["alternate", "diversion airport", "diversion aerodrome"],
    "runway_contamination": ["braking action", "runway condition", "rwycc", "contaminated runway", "slush", "snow on the runway"],
    "dispatch_release": ["dispatch release", "flight release", "flight plan", "reissue"],
    "fdtl": ["duty time", "flight duty period", "fdp", "rest period", "crew rest"],
    "mel": ["mel", "minimum equipment list", "inoperative", "placard"],
    "convective": ["thunderstorm", "convective", "lightning", "turbulence"],
    "low_visibility": ["cat ii", "cat iii", "cat 2", "cat 3", "autoland", "low visibility"],
    "emergency_fuel": ["minimum fuel", "mayday fuel", "fuel declaration"],
}

# Generic "N <unit>" numeric-entity patterns (capacity, thresholds, counts).
_NUMERIC_ENTITY_RE = re.compile(
    r"\b\d+(\.\d+)?\s*"
    r"(people|passengers|knots|kts|minutes|min|hours|hrs|feet|ft|meters|metres|m|"
    r"nm|nautical miles|degrees|percent|%|days)\b",
    re.IGNORECASE,
)

# Capitalized multi-word proper-noun-ish spans (place names, aircraft types
# like "A320" or "B737-800"). Sentence-initial interrogatives/auxiliaries
# ("What", "Do", "Is", ...) are excluded via _WH_STOPWORDS below since they
# capitalize for grammar reasons, not because they name an entity.
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][A-Za-z0-9]{1,}(?:-[A-Za-z0-9]+)?)\b")

_WH_STOPWORDS = {
    "what", "how", "when", "where", "why", "who", "which", "do", "does",
    "is", "are", "can", "could", "should", "would", "will", "the", "this",
    "that", "please", "summarize", "i", "we",
}


def is_meaningful_proper_noun(word: str) -> bool:
    return word.lower() not in _WH_STOPWORDS


# Public aliases for consumers outside this module (decomposer.py) that
# want the same entity patterns for context-carrying, not just detection.
NUMERIC_ENTITY_RE = _NUMERIC_ENTITY_RE
PROPER_NOUN_RE = _PROPER_NOUN_RE


def find_topics(text: str) -> set[str]:
    lower = text.lower()
    hits = set()
    for topic, kws in TOPIC_KEYWORDS.items():
        if any(kw in lower for kw in kws):
            hits.add(topic)
    return hits


def has_stable_entity(text: str) -> bool:
    """True if the text contains a concrete, retrieval-worthy entity: a
    known topic keyword, a numeric-unit quantity, or a proper noun that
    isn't just the sentence-initial word."""

    if find_topics(text):
        return True
    if _NUMERIC_ENTITY_RE.search(text):
        return True
    words = text.strip().split()
    for i, w in enumerate(words):
        if i == 0:
            continue  # sentence-initial capitalisation doesn't count
        cleaned = w.strip(",.;:()")
        if _PROPER_NOUN_RE.fullmatch(cleaned) and is_meaningful_proper_noun(cleaned):
            return True
    return False
