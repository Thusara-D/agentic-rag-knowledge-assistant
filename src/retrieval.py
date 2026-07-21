"""Retrieval functions will be implemented in Milestone 3."""

from hashlib import sha256
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


DEFAULT_DATABASE_PATH = Path("chroma_db")
DEFAULT_COLLECTION_NAME = "knowledge_base"


def get_vector_collection(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Any:
    """Create or load the persistent Chroma collection."""

    if not collection_name.strip():
        raise ValueError("Collection name cannot be empty.")

    database_directory = Path(database_path)
    database_directory.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(database_directory),
    )

    embedding_function = DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        configuration={
            "hnsw": {
                "space": "cosine",
            }
        },
    )

    return collection


def create_chunk_id(
    source: str,
    chunk_index: int,
) -> str:
    """Create a stable unique ID for one document chunk."""

    raw_id = f"{source}:{chunk_index}"

    return sha256(
        raw_id.encode("utf-8")
    ).hexdigest()


def store_chunks(
    collection: Any,
    chunks: list[str],
    source: str,
) -> int:
    """Create embeddings and store document chunks in Chroma."""

    if not source.strip():
        raise ValueError("Source name cannot be empty.")

    cleaned_chunks = [
        chunk.strip()
        for chunk in chunks
        if chunk and chunk.strip()
    ]

    if not cleaned_chunks:
        raise ValueError("At least one non-empty chunk is required.")

    chunk_ids = [
        create_chunk_id(
            source=source,
            chunk_index=index,
        )
        for index in range(len(cleaned_chunks))
    ]

    metadata = [
        {
            "source": source,
            "chunk_index": index,
        }
        for index in range(len(cleaned_chunks))
    ]

    collection.upsert(
        ids=chunk_ids,
        documents=cleaned_chunks,
        metadatas=metadata,
    )

    return len(cleaned_chunks)


def search_similar_chunks(
    collection: Any,
    question: str,
    number_of_results: int = 3,
) -> list[dict[str, Any]]:
    """Retrieve chunks whose meanings are closest to the question."""

    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    if number_of_results <= 0:
        raise ValueError(
            "number_of_results must be greater than zero."
        )

    stored_chunk_count = collection.count()

    if stored_chunk_count == 0:
        return []

    result_limit = min(
        number_of_results,
        stored_chunk_count,
    )

    results = collection.query(
        query_texts=[cleaned_question],
        n_results=result_limit,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results["documents"][0]
    metadata_items = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved_chunks: list[dict[str, Any]] = []

    for document, metadata, distance in zip(
        documents,
        metadata_items,
        distances,
    ):
        similarity = 1 - float(distance)

        retrieved_chunks.append(
            {
                "text": document,
                "source": metadata["source"],
                "chunk_index": metadata["chunk_index"],
                "distance": float(distance),
                "similarity": similarity,
            }
        )

    return retrieved_chunks