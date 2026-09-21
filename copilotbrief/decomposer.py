"""Component 2 — Multi-Intent Decomposer.

Parses a compound, unsegmented utterance into discrete, search-ready
sub-queries. Two failure modes called out explicitly in the Theme 4 guide
drive the design here:

  - "Over-Fragmenting Sub-Queries": splitting a single simple question into
    multiple near-identical queries pollutes the reranker. We enforce a
    minimum content-token count per fragment and cap sub-query count.
  - Losing shared context: a location, aircraft type, or other entity
    mentioned once at the top of the utterance is implicitly still in scope
    for later clauses ("the crosswind limit, whether we need de-icing, and
    the diversion fuel reserve" are all about *this aircraft*). We carry
    forward entities from earlier clauses into later ones that lack their
    own, rather than searching each clause in total isolation.
"""

from __future__ import annotations

import re

from . import clausing, config, taxonomy
from .models import SubQuery
from .retrieval import tokenize

_UTTERANCE_END_RE = re.compile(r"\[utterance end\]", re.IGNORECASE)


def _split_clauses(text: str) -> list[str]:
    text = _UTTERANCE_END_RE.sub("", text)
    clauses = clausing.split_clauses(text, min_tokens=1)  # filtering by real
    # min-token threshold happens in decompose() below, after we know
    # config.MIN_SUBQUERY_TOKENS; keep every non-empty fragment here.
    if not clauses:
        cleaned = clausing.clean_clause(text)
        clauses = [cleaned] if cleaned else []
    return clauses


def _shared_context_entities(clauses: list[str]) -> list[str]:
    """Entities (topic keywords, numeric quantities, proper nouns) found in
    the FIRST clause only — candidates to carry forward into later clauses
    that don't mention their own."""

    if not clauses:
        return []
    anchor = clauses[0]
    entities: list[str] = []
    for m in taxonomy.NUMERIC_ENTITY_RE.finditer(anchor):
        entities.append(m.group(0))
    for m in taxonomy.PROPER_NOUN_RE.finditer(anchor):
        word = m.group(0)
        if taxonomy.is_meaningful_proper_noun(word) and word not in entities:
            entities.append(word)
    return entities


class MultiIntentDecomposer:
    """Stateless: pure function of the accumulated utterance text."""

    def decompose(self, accumulated_text: str) -> list[SubQuery]:
        clauses = _split_clauses(accumulated_text)

        # Filter out fragments too short to be a standalone, search-worthy
        # sub-query (the over-fragmentation guard).
        clauses = [c for c in clauses if len(tokenize(c)) >= config.MIN_SUBQUERY_TOKENS]
        if not clauses:
            cleaned = _clean(accumulated_text)
            clauses = [cleaned] if cleaned else []
        clauses = clauses[: config.MAX_SUBQUERIES]

        shared_entities = _shared_context_entities(clauses)

        sub_queries: list[SubQuery] = []
        for i, clause in enumerate(clauses):
            enriched = clause
            if i > 0:
                # Carry forward any shared entity from the anchor clause
                # that this later clause doesn't already mention itself,
                # so "the cancellation policy" doesn't get searched with
                # zero knowledge of the venue/aircraft it's about.
                missing = [
                    e for e in shared_entities
                    if e.lower() not in clause.lower()
                ]
                if missing:
                    enriched = f"{clause} ({', '.join(missing)})"
            topics = taxonomy.find_topics(enriched)
            topic_hint = sorted(topics)[0] if topics else None
            sub_queries.append(SubQuery(text=enriched, topic_hint=topic_hint))

        return sub_queries
