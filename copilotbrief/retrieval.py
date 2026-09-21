"""Hybrid (dense + sparse) retrieval with Reciprocal Rank Fusion.

Component 3 of the pipeline: "Corpus Retrieval & Fusion — Search Supplied
Corpus, Dense/Sparse Hybrid Scoring, Re-rank & Deduplicate Chunks".

Design note on reproducibility (Gate G1, "container launches via a single
command on a clean machine"): the dense encoder (sentence-transformers)
needs a one-time model download. If that download is unavailable at
runtime (offline grading box, sandboxed network), the retriever
transparently falls back to a TF-IDF vector space as the dense-analogue
signal instead of hard-failing. Either way the pipeline still performs a
genuine hybrid (two independent scoring signals fused by rank), it just
degrades the *source* of the second signal rather than the architecture.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .models import CorpusChunk, ScoredChunk

logger = logging.getLogger("copilotbrief.retrieval")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class _DenseBackend:
    """Wraps whichever dense-similarity backend is actually available."""

    def __init__(self, texts: list[str]):
        self.texts = texts
        self.backend_name = "unavailable"
        self._model = None
        self._chunk_vectors = None
        self._tfidf = None
        self._init_sentence_transformers(texts)
        if self._model is None:
            self._init_tfidf(texts)

    def _init_sentence_transformers(self, texts: list[str]) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            try:
                # Fast path: load strictly from local cache, no network call.
                # This is what a Docker image pre-warmed at build time hits.
                model = SentenceTransformer(config.DENSE_MODEL_NAME, local_files_only=True)
            except Exception:
                if not config.ALLOW_DENSE_MODEL_DOWNLOAD:
                    # Not cached, and we were not explicitly told to hit the
                    # network (e.g. we are not inside the Docker build step
                    # that pre-warms the cache). Skip straight to TF-IDF
                    # rather than retrying network calls for ~30s.
                    raise
                model = SentenceTransformer(config.DENSE_MODEL_NAME)

            vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            self._model = model
            self._chunk_vectors = np.asarray(vectors)
            self.backend_name = "sentence-transformers"
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure -> fallback
            logger.warning("Dense encoder unavailable (%s); falling back to TF-IDF.", exc)
            self._model = None

    def _init_tfidf(self, texts: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False)
        self._chunk_vectors = self._tfidf.fit_transform(texts)
        self.backend_name = "tfidf"

    def query_scores(self, query: str) -> np.ndarray:
        if self.backend_name == "sentence-transformers":
            qvec = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            return self._chunk_vectors @ qvec
        # TF-IDF cosine similarity
        qvec = self._tfidf.transform([query])
        sims = (self._chunk_vectors @ qvec.T).toarray().ravel()
        return sims


@dataclass
class _ChunkIndexEntry:
    chunk: CorpusChunk
    tokens: list[str]


class HybridRetriever:
    """BM25 (sparse) + sentence-embedding-or-TF-IDF (dense) fused by RRF."""

    def __init__(self, chunks: list[CorpusChunk], use_dense: bool = True):
        if not chunks:
            raise ValueError("HybridRetriever requires a non-empty chunk list.")
        self.chunks = chunks
        self.use_dense = use_dense
        texts = [c.text for c in chunks]

        tokenized = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        # use_dense=False gives a pure-BM25 retriever — used by the ablation
        # study in benchmark/evaluate.py to quantify what the dense/TF-IDF
        # fusion signal actually contributes over sparse-only retrieval.
        self._dense = _DenseBackend(texts) if use_dense else None

    @property
    def dense_backend_name(self) -> str:
        return self._dense.backend_name if self._dense is not None else "disabled (sparse-only)"

    def _rrf_fuse(self, rank_lists: list[list[int]], k: int = config.RRF_K) -> dict[int, float]:
        fused: dict[int, float] = {}
        for ranks in rank_lists:
            for rank, idx in enumerate(ranks):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return fused

    def search(self, query: str, top_k: int = config.TOP_K_PER_SUBQUERY) -> list[ScoredChunk]:
        query = query.strip()
        if not query:
            return []

        bm25_scores = self._bm25.get_scores(tokenize(query))
        bm25_ranks = list(np.argsort(-bm25_scores))

        rank_lists = [bm25_ranks]
        if self._dense is not None:
            dense_scores = self._dense.query_scores(query)
            rank_lists.append(list(np.argsort(-dense_scores)))

        fused = self._rrf_fuse(rank_lists)
        ranked_idx = sorted(fused.keys(), key=lambda i: -fused[i])[:top_k]

        results = [
            ScoredChunk(chunk=self.chunks[i], score=fused[i], sub_query=query)
            for i in ranked_idx
        ]
        return results

    def search_many(
        self, queries: list[str], top_k_per_query: int = config.TOP_K_PER_SUBQUERY
    ) -> list[ScoredChunk]:
        """Run retrieval for several sub-queries and merge/deduplicate by (doc_id, section_id),
        keeping the highest score seen for each chunk. This is the "Evidence Fusion &
        Reranking" step: collects candidates across all sub-queries, reconciles
        redundant evidence."""

        best: dict[tuple[str, str], ScoredChunk] = {}
        for q in queries:
            for sc in self.search(q, top_k=top_k_per_query):
                key = (sc.chunk.doc_id, sc.chunk.section_id)
                if key not in best or sc.score > best[key].score:
                    best[key] = sc
        return sorted(best.values(), key=lambda sc: -sc.score)
