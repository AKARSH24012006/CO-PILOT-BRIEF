import logging

import pytest

from copilotbrief.corpus import load_corpus
from copilotbrief.pipeline import CopilotBriefEngine
from copilotbrief.retrieval import HybridRetriever

logging.getLogger("copilotbrief.retrieval").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


@pytest.fixture(scope="session")
def corpus_chunks():
    return load_corpus()


@pytest.fixture(scope="session")
def retriever(corpus_chunks):
    return HybridRetriever(corpus_chunks)


@pytest.fixture
def engine():
    # Fresh engine per test so session state never leaks between tests.
    return CopilotBriefEngine()
