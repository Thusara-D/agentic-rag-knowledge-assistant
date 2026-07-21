from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import chromadb
import pytest
from chromadb.api.types import (
    Documents,
    EmbeddingFunction,
    Embeddings,
)

from src.retrieval import (
    create_chunk_id,
    search_similar_chunks,
    store_chunks,
)


class FakeEmbeddingFunction(EmbeddingFunction[Documents]):
    """Create simple predictable vectors for automated tests."""

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(text) for text in input]

    @staticmethod
    def name() -> str:
        return "fake_embedding"

    @staticmethod
    def build_from_config(
        config: dict[str, Any],
    ) -> "FakeEmbeddingFunction":
        return FakeEmbeddingFunction()

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def _embed(text: str) -> list[float]:
        cleaned_text = text.lower()

        if any(
            word in cleaned_text
            for word in ["mfa", "authentication", "login"]
        ):
            return [1.0, 0.0, 0.0]

        if "password" in cleaned_text:
            return [0.8, 0.2, 0.0]

        if any(
            word in cleaned_text
            for word in ["leave", "holiday"]
        ):
            return [0.0, 0.0, 1.0]

        return [0.0, 1.0, 0.0]


@pytest.fixture
def test_collection() -> Iterator[Any]:
    """Create a separate in-memory Chroma collection for each test."""

    client = chromadb.EphemeralClient()
    collection_name = f"retrieval_test_{uuid4().hex}"

    collection = client.create_collection(
        name=collection_name,
        embedding_function=FakeEmbeddingFunction(),
        configuration={
            "hnsw": {
                "space": "cosine",
            }
        },
    )

    try:
        yield collection
    finally:
        client.delete_collection(collection_name)


def test_create_chunk_id_is_stable():
    """The same source and index should produce the same ID."""

    first_id = create_chunk_id(
        source="security_policy.pdf",
        chunk_index=0,
    )

    second_id = create_chunk_id(
        source="security_policy.pdf",
        chunk_index=0,
    )

    assert first_id == second_id


def test_create_chunk_id_changes_for_different_chunks():
    """Different chunk indexes should produce different IDs."""

    first_id = create_chunk_id(
        source="security_policy.pdf",
        chunk_index=0,
    )

    second_id = create_chunk_id(
        source="security_policy.pdf",
        chunk_index=1,
    )

    assert first_id != second_id


def test_store_chunks_adds_documents(test_collection: Any):
    """Valid chunks should be stored in Chroma."""

    chunks = [
        "Employees must use multi-factor authentication.",
        "Passwords must not be shared.",
    ]

    stored_count = store_chunks(
        collection=test_collection,
        chunks=chunks,
        source="security_policy.txt",
    )

    assert stored_count == 2
    assert test_collection.count() == 2


def test_store_chunks_rejects_empty_chunks(test_collection: Any):
    """An empty chunk list should raise an error."""

    with pytest.raises(
        ValueError,
        match="At least one non-empty chunk",
    ):
        store_chunks(
            collection=test_collection,
            chunks=[],
            source="security_policy.txt",
        )


def test_search_returns_relevant_chunk_first(
    test_collection: Any,
):
    """Semantic search should return the MFA chunk first."""

    chunks = [
        "Employees must use multi-factor authentication.",
        "Passwords must not be shared.",
        "Employees receive twenty days of annual leave.",
    ]

    store_chunks(
        collection=test_collection,
        chunks=chunks,
        source="company_policy.txt",
    )

    results = search_similar_chunks(
        collection=test_collection,
        question="How should staff protect their login?",
        number_of_results=3,
    )

    assert len(results) == 3
    assert "multi-factor authentication" in results[0]["text"]
    assert results[0]["source"] == "company_policy.txt"
    assert results[0]["chunk_index"] == 0


def test_search_empty_collection_returns_empty(
    test_collection: Any,
):
    """Searching an empty collection should return an empty list."""

    results = search_similar_chunks(
        collection=test_collection,
        question="What is the authentication policy?",
    )

    assert results == []


def test_search_rejects_empty_question(
    test_collection: Any,
):
    """An empty question should raise an error."""

    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        search_similar_chunks(
            collection=test_collection,
            question="   ",
        )