from pathlib import Path

import streamlit as st

from src.ingestion import chunk_text, load_document
from src.retrieval import (
    get_vector_collection,
    search_similar_chunks,
    store_chunks,
)


UPLOAD_DIRECTORY = Path("uploads")


@st.cache_resource
def load_vector_collection():
    """Create the Chroma collection once and reuse it."""
    return get_vector_collection()


def save_uploaded_file(uploaded_file) -> Path:
    """Save a Streamlit uploaded file inside the uploads folder."""

    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(uploaded_file.name).name
    saved_path = UPLOAD_DIRECTORY / safe_filename

    saved_path.write_bytes(uploaded_file.getbuffer())

    return saved_path


st.set_page_config(
    page_title="Agentic RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("Agentic RAG Knowledge Assistant")
st.caption(
    "Upload documents and retrieve relevant evidence using semantic search."
)

collection = load_vector_collection()

st.subheader("1. Upload a document")

uploaded_file = st.file_uploader(
    "Select a TXT or PDF file",
    type=["txt", "pdf"],
)

if uploaded_file is not None:
    st.write(f"Selected file: **{uploaded_file.name}**")

    if st.button("Process document", type="primary"):
        try:
            with st.spinner("Processing document..."):
                saved_file_path = save_uploaded_file(uploaded_file)

                document_text = load_document(saved_file_path)

                chunks = chunk_text(
                    document_text,
                    chunk_size=500,
                    chunk_overlap=50,
                )

                stored_count = store_chunks(
                    collection=collection,
                    chunks=chunks,
                    source=uploaded_file.name,
                )

            st.success(
                f"Document processed successfully. "
                f"{stored_count} chunks were stored."
            )

        except Exception as error:
            st.error(f"Document processing failed: {error}")

st.divider()

st.subheader("2. Search the knowledge base")

question = st.text_input(
    "Ask a question about the uploaded documents",
    placeholder="Example: How should employees protect their accounts?",
)

number_of_results = st.slider(
    "Number of evidence chunks",
    min_value=1,
    max_value=5,
    value=3,
)

if st.button("Search knowledge base"):
    if not question.strip():
        st.warning("Enter a question before searching.")

    else:
        try:
            with st.spinner("Searching for relevant evidence..."):
                results = search_similar_chunks(
                    collection=collection,
                    question=question,
                    number_of_results=number_of_results,
                )

            if not results:
                st.warning(
                    "The knowledge base is empty. "
                    "Upload and process a document first."
                )

            else:
                st.success(
                    f"Found {len(results)} relevant evidence chunks."
                )

                for position, result in enumerate(
                    results,
                    start=1,
                ):
                    source = result["source"]
                    chunk_number = result["chunk_index"] + 1
                    similarity = result["similarity"]

                    with st.expander(
                        f"Result {position}: {source} "
                        f"— chunk {chunk_number}"
                    ):
                        st.write(result["text"])
                        st.write(
                            f"**Similarity score:** "
                            f"{similarity:.3f}"
                        )

        except Exception as error:
            st.error(f"Search failed: {error}")

st.divider()

st.caption(
    f"Stored chunks currently available: {collection.count()}"
)