"""Shared clause-splitting used by both controller.py (to COUNT distinct
sub-intents for the Wait/Retrieve decision) and decomposer.py (to actually
BUILD the sub-queries). Kept in one place so the two components can never
disagree on what counts as a clause boundary — if the controller decided
"3 sub-intents" fired a MULTI_INTENT_RETRIEVE, the decomposer needs to
produce the same 3 clauses moments later.

Two kinds of boundary are recognized:
  1. Coordinating "list glue": and / also / plus / as well as.
  2. A comma immediately followed by a WH-question opener (whether, what,
     do we, is there, ...) — covers Oxford-comma-style compound questions
     ("the crosswind limit, whether we need de-icing, and the fuel
     reserve") where only the LAST item is glued with "and".
"""

from __future__ import annotations

import re

from .retrieval import tokenize

_AND_GLUE_RE = re.compile(r"\band\b|\balso\b|\bas well as\b|\bplus\b", re.IGNORECASE)
_LIST_SPLIT_RE = re.compile(r"\band\b|\balso\b|\bas well as\b|\bplus\b|,", re.IGNORECASE)

_FILLER_LEAD_RE = re.compile(
    r"^(and\s+)?(before\s+we\s+\w+[\w\s]{0,20}?,\s*)?i\s+(also\s+)?"
    r"(need\s+to\s+know|need|want|would\s+like|have\s+to|must|also\s+need)\s*",
    re.IGNORECASE,
)

MIN_CLAUSE_TOKENS = 2


def clean_clause(text: str) -> str:
    text = text.strip()
    text = _FILLER_LEAD_RE.sub("", text).strip(" ,.;:")
    return text


def raw_split(text: str) -> list[str]:
    """A bare comma is only treated as a clause boundary once the text
    ALSO contains an "and"/"also"/"plus" glue word somewhere — that's
    independent evidence this IS a list (Oxford-comma style: "X, Y, and
    Z" — only the last item is glued with "and", but every comma before
    it is a list separator too). Without an "and" anywhere, a comma is
    just ordinary sentence punctuation ("For dispatch planning, what's
    the fuel rule?" is ONE question, not two), and splitting on it would
    over-fragment a simple single-intent query — the "Over-Fragmenting
    Sub-Queries" pitfall the spec warns about."""

    has_list_glue = bool(_AND_GLUE_RE.search(text))
    pattern = _LIST_SPLIT_RE if has_list_glue else _AND_GLUE_RE
    return pattern.split(text)


def split_clauses(text: str, min_tokens: int = MIN_CLAUSE_TOKENS) -> list[str]:
    """Return cleaned, content-bearing clauses (the over-fragmentation
    guard: anything left with fewer than `min_tokens` content tokens after
    filler-stripping is dropped, e.g. a dangling "I need" fragment).

    The leading filler/intro ("Before we release this flight, I need to
    know...") is stripped from the WHOLE text before splitting, not after
    — otherwise an intro's own comma ("Before we release this flight, I
    need to know...") gets mistaken for a list-item boundary once list
    splitting is active, leaving a spurious "Before we release this
    flight" fragment behind."""

    text = _FILLER_LEAD_RE.sub("", text.strip())
    clauses = []
    for part in raw_split(text):
        cleaned = clean_clause(part)
        if cleaned and len(tokenize(cleaned)) >= min_tokens:
            clauses.append(cleaned)
    return clauses
