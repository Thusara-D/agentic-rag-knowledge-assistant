from pathlib import Path

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def validate_document(file_path: str | Path) -> Path:
    """Validate that the document exists and has a supported file type."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            "Only TXT and PDF files are supported."
        )

    return path


def load_text_file(file_path: str | Path) -> str:
    """Read and return text from a UTF-8 TXT file."""
    path = validate_document(file_path)

    if path.suffix.lower() != ".txt":
        raise ValueError("The provided document is not a TXT file.")

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError("The TXT file does not contain readable text.")

    return text


def load_pdf_file(file_path: str | Path) -> str:
    """Extract and return text from a text-based PDF file."""
    path = validate_document(file_path)

    if path.suffix.lower() != ".pdf":
        raise ValueError("The provided document is not a PDF file.")

    reader = PdfReader(str(path))
    extracted_pages: list[str] = []

    for page in reader.pages:
        page_text = page.extract_text() or ""

        if page_text.strip():
            extracted_pages.append(page_text.strip())

    if not extracted_pages:
        raise ValueError(
            "The PDF does not contain readable text. "
            "It may be a scanned image PDF."
        )

    return "\n\n".join(extracted_pages)


def load_document(file_path: str | Path) -> str:
    """Load a supported document based on its file extension."""
    path = validate_document(file_path)

    if path.suffix.lower() == ".txt":
        return load_text_file(path)

    if path.suffix.lower() == ".pdf":
        return load_pdf_file(path)

    raise ValueError(f"Unsupported file type: {path.suffix}")

def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Split text into overlapping chunks."""

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    cleaned_text = " ".join(text.split())
    chunks: list[str] = []

    start = 0

    while start < len(cleaned_text):
        end = start + chunk_size
        chunk = cleaned_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - chunk_overlap

    return chunks