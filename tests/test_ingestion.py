from src.ingestion import load_text_file


def test_load_text_file(tmp_path):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("Agentic RAG test document", encoding="utf-8")

    content = load_text_file(sample_file)

    assert content == "Agentic RAG test document"
