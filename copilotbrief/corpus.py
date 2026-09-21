"""Corpus loading and section-aware chunking.

Each corpus file is a markdown document with:
    # Doc_ID — Title
    ## §N Section Title
    body text...

Chunks are built per-section (splitting further only if a section exceeds
CHUNK_MAX_WORDS) so every citation stays at the granularity the Theme 4
spec requires: "[Doc_ID §Section]".
"""

from __future__ import annotations

import re
from pathlib import Path

from . import config
from .models import CorpusChunk

_DOC_ID_RE = re.compile(r"^#\s+(?P<doc_id>\S+)\s*[—-]\s*(?P<title>.+)$", re.MULTILINE)
_SECTION_RE = re.compile(
    r"^##\s*§(?P<section_id>[\w.]+)\s+(?P<section_title>.+)$", re.MULTILINE
)


def _split_words(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    parts = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        parts.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return parts


def parse_document(raw_text: str, source_name: str) -> list[CorpusChunk]:
    doc_match = _DOC_ID_RE.search(raw_text)
    doc_id = doc_match.group("doc_id") if doc_match else Path(source_name).stem

    section_matches = list(_SECTION_RE.finditer(raw_text))
    chunks: list[CorpusChunk] = []
    for i, m in enumerate(section_matches):
        section_id = m.group("section_id")
        section_title = m.group("section_title").strip()
        body_start = m.end()
        body_end = section_matches[i + 1].start() if i + 1 < len(section_matches) else len(raw_text)
        body = raw_text[body_start:body_end].strip()
        if not body:
            continue
        parts = _split_words(body, config.CHUNK_MAX_WORDS, config.CHUNK_OVERLAP_WORDS)
        for idx, part in enumerate(parts):
            chunks.append(
                CorpusChunk(
                    doc_id=doc_id,
                    section_id=section_id,
                    section_title=section_title,
                    text=part,
                    chunk_index=idx,
                )
            )
    return chunks


def load_corpus(corpus_dir: Path | None = None) -> list[CorpusChunk]:
    corpus_dir = Path(corpus_dir or config.CORPUS_DIR)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    all_chunks: list[CorpusChunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        all_chunks.extend(parse_document(raw, path.name))

    if not all_chunks:
        raise ValueError(f"No chunks parsed from corpus at {corpus_dir}")
    return all_chunks


def load_corpus_catalog(corpus_dir: Path | None = None) -> list[dict]:
    """Lightweight doc/section TITLE catalog (no chunk text) for driving the
    Evidence panel and citation tooltips in the UI — the pipeline already
    parses these headers per-chunk in `parse_document`, but discards the
    title strings after chunking; this walks the same regexes again purely
    to expose `{doc_id, title, sections: [{section_id, section_title}]}` so
    the frontend can resolve a bare "[Doc_03 §2]" citation into something a
    human can read without re-shipping full chunk text over the wire."""

    corpus_dir = Path(corpus_dir or config.CORPUS_DIR)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    catalog: list[dict] = []
    for path in sorted(corpus_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        doc_match = _DOC_ID_RE.search(raw)
        doc_id = doc_match.group("doc_id") if doc_match else Path(path.name).stem
        title = doc_match.group("title").strip() if doc_match else doc_id
        sections = [
            {"section_id": m.group("section_id"), "section_title": m.group("section_title").strip()}
            for m in _SECTION_RE.finditer(raw)
        ]
        catalog.append({"doc_id": doc_id, "title": title, "sections": sections})
    return catalog
