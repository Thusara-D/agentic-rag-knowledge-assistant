from pathlib import Path


def load_text_file(file_path: str | Path) -> str:
    """Load a UTF-8 text file.

    PDF loading and chunking will be added in Milestone 2.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if path.suffix.lower() != ".txt":
        raise ValueError("Milestone 1 currently supports only .txt files.")

    return path.read_text(encoding="utf-8")
