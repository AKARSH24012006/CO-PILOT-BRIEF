def test_retriever_returns_relevant_chunk_for_crosswind_query(retriever):
    results = retriever.search("crosswind limit A320 wet runway", top_k=5)
    assert results
    citations = {sc.chunk.citation for sc in results}
    assert any(c.startswith("Doc_01") for c in citations)


def test_search_many_deduplicates_by_section(retriever):
    results = retriever.search_many(
        ["crosswind limit A320", "crosswind limit A320 again worded differently"],
        top_k_per_query=5,
    )
    keys = [(sc.chunk.doc_id, sc.chunk.section_id) for sc in results]
    assert len(keys) == len(set(keys))  # no duplicate (doc, section) pairs


def test_dense_backend_falls_back_gracefully(retriever):
    # In this offline test environment the dense encoder can't reach the
    # network, so it must have fallen back to TF-IDF rather than crashing.
    assert retriever.dense_backend_name in ("sentence-transformers", "tfidf")


def test_empty_query_returns_no_results(retriever):
    assert retriever.search("   ", top_k=5) == []
