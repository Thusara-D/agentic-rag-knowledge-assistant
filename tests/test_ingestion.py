from pathlib import Path

import pytest

from src.ingestion import (
    chunk_text,
    load_document,
    load_text_file,
    validate_document,
)


def test_validate_existing_txt_file(tmp_path: Path):
    """A valid TXT file should pass validation."""
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("Test content", encoding="utf-8")

    validated_path = validate_document(sample_file)

    assert validated_path == sample_file


def test_load_text_file(tmp_path: Path):
    """The loader should return the text inside a TXT file."""
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text(
        "Agentic RAG test document",
        encoding="utf-8",
    )

    content = load_text_file(sample_file)

    assert content == "Agentic RAG test document"


def test_load_document_automatically_loads_txt(tmp_path: Path):
    """load_document should select the TXT loader automatically."""
    sample_file = tmp_path / "policy.txt"
    sample_file.write_text(
        "Employees must use multi-factor authentication.",
        encoding="utf-8",
    )

    content = load_document(sample_file)

    assert content == "Employees must use multi-factor authentication."


def test_missing_document_raises_error(tmp_path: Path):
    """A missing document should raise FileNotFoundError."""
    missing_file = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        validate_document(missing_file)


def test_unsupported_file_type_raises_error(tmp_path: Path):
    """Unsupported file types should be rejected."""
    image_file = tmp_path / "image.jpg"
    image_file.write_text("Fake image content", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        validate_document(image_file)


def test_empty_text_file_raises_error(tmp_path: Path):
    """An empty TXT file should not be accepted."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="does not contain readable text",
    ):
        load_text_file(empty_file)

def test_chunk_text_creates_multiple_chunks():
    """Long text should be divided into smaller chunks."""
    text = "A" * 1200

    chunks = chunk_text(
        text,
        chunk_size=500,
        chunk_overlap=50,
    )

    assert len(chunks) == 3
    assert len(chunks[0]) == 500
    assert len(chunks[1]) == 500


def test_chunk_text_keeps_short_text_in_one_chunk():
    """Short text should remain as one chunk."""
    text = "Employees must use multi-factor authentication."

    chunks = chunk_text(text)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_rejects_empty_text():
    """Empty text should raise an error."""
    with pytest.raises(ValueError, match="Text cannot be empty"):
        chunk_text("")


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    """Overlap cannot be equal to or larger than chunk size."""
    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller",
    ):
        chunk_text(
            "Example text",
            chunk_size=100,
            chunk_overlap=100,
        )