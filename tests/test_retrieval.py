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
    clear_knowledge_base,
    create_chunk_id,
    delete_indexed_document,
    list_indexed_documents,
    search_similar_chunks,
    store_chunks,
)


class FakeEmbeddingFunction(EmbeddingFunction[Documents]):
    """Create simple, predictable vectors for automated tests."""

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        """Create an embedding for every supplied text item."""
        return [self._embed(text) for text in input]

    @staticmethod
    def name() -> str:
        """Return a name for the custom embedding function."""
        return "fake_embedding"

    @staticmethod
    def build_from_config(
        config: dict[str, Any],
    ) -> "FakeEmbeddingFunction":
        """Create the embedding function from configuration."""
        return FakeEmbeddingFunction()

    def get_config(self) -> dict[str, Any]:
        """Return the embedding function configuration."""
        return {}

    @staticmethod
    def _embed(text: str) -> list[float]:
        """Convert text into a predictable three-dimensional vector."""
        cleaned_text = text.lower()

        if any(
            word in cleaned_text
            for word in [
                "mfa",
                "authentication",
                "login",
            ]
        ):
            return [1.0, 0.0, 0.0]

        if "password" in cleaned_text:
            return [0.8, 0.2, 0.0]

        if any(
            word in cleaned_text
            for word in [
                "leave",
                "holiday",
            ]
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


def test_store_chunks_adds_documents(
    test_collection: Any,
):
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


def test_store_chunks_rejects_empty_chunks(
    test_collection: Any,
):
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


def test_list_indexed_documents_returns_chunk_counts(
    test_collection: Any,
):
    """Indexed documents should include their stored chunk counts."""
    store_chunks(
        collection=test_collection,
        chunks=[
            "First security policy chunk.",
            "Second security policy chunk.",
        ],
        source="security_policy.txt",
    )

    store_chunks(
        collection=test_collection,
        chunks=[
            "Employee leave policy.",
        ],
        source="leave_policy.txt",
    )

    documents = list_indexed_documents(
        test_collection
    )

    assert documents == [
        {
            "source": "leave_policy.txt",
            "chunk_count": 1,
        },
        {
            "source": "security_policy.txt",
            "chunk_count": 2,
        },
    ]


def test_list_indexed_documents_returns_empty_list(
    test_collection: Any,
):
    """An empty collection should contain no indexed documents."""
    documents = list_indexed_documents(
        test_collection
    )

    assert documents == []


def test_delete_indexed_document_removes_only_selected_file(
    test_collection: Any,
):
    """Deleting one document should leave other documents unchanged."""
    store_chunks(
        collection=test_collection,
        chunks=[
            "Employees must use MFA.",
            "Passwords must not be shared.",
        ],
        source="security_policy.txt",
    )

    store_chunks(
        collection=test_collection,
        chunks=[
            "Employees receive annual leave.",
        ],
        source="leave_policy.txt",
    )

    deleted_count = delete_indexed_document(
        collection=test_collection,
        source="security_policy.txt",
    )

    remaining_documents = list_indexed_documents(
        test_collection
    )

    assert deleted_count == 2
    assert test_collection.count() == 1

    assert remaining_documents == [
        {
            "source": "leave_policy.txt",
            "chunk_count": 1,
        }
    ]


def test_delete_indexed_document_rejects_empty_source(
    test_collection: Any,
):
    """A blank source name should raise an error."""
    with pytest.raises(
        ValueError,
        match="Source name cannot be empty",
    ):
        delete_indexed_document(
            collection=test_collection,
            source="   ",
        )


def test_clear_knowledge_base_removes_all_chunks(
    test_collection: Any,
):
    """Clearing the knowledge base should remove every stored chunk."""
    store_chunks(
        collection=test_collection,
        chunks=[
            "Employees must use MFA.",
            "Passwords must not be shared.",
        ],
        source="security_policy.txt",
    )

    store_chunks(
        collection=test_collection,
        chunks=[
            "Employees receive annual leave.",
        ],
        source="leave_policy.txt",
    )

    deleted_count = clear_knowledge_base(
        test_collection
    )

    assert deleted_count == 3
    assert test_collection.count() == 0
    assert list_indexed_documents(test_collection) == []