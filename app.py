from pathlib import Path

import streamlit as st

from src.graph import build_agent_graph
from src.ingestion import chunk_text, load_document

from src.retrieval import (
    clear_knowledge_base,
    delete_indexed_document,
    get_vector_collection,
    list_indexed_documents,
    store_chunks,
)


UPLOAD_DIRECTORY = Path("uploads")


@st.cache_resource
def load_application_resources():
    """Create the Chroma collection and LangGraph workflow once."""

    collection = get_vector_collection()
    agent_graph = build_agent_graph(collection)

    return collection, agent_graph


def save_uploaded_file(uploaded_file) -> Path:
    """Save an uploaded file inside the local uploads folder."""

    UPLOAD_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_filename = Path(uploaded_file.name).name
    saved_path = UPLOAD_DIRECTORY / safe_filename

    saved_path.write_bytes(
        uploaded_file.getbuffer()
    )

    return saved_path


st.set_page_config(
    page_title="Agentic RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("Agentic RAG Knowledge Assistant")

st.caption(
    "Upload documents and receive evidence-grounded answers "
    "that are automatically verified and corrected."
)

collection, agent_graph = load_application_resources()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "sidebar_message" not in st.session_state:
    st.session_state.sidebar_message = ""

with st.sidebar:
    st.header("Knowledge Base")

    if st.session_state.sidebar_message:
        st.success(
            st.session_state.sidebar_message
        )
        st.session_state.sidebar_message = ""

    indexed_documents = list_indexed_documents(
        collection
    )

    column_one, column_two = st.columns(2)

    with column_one:
        st.metric(
            "Documents",
            len(indexed_documents),
        )

    with column_two:
        st.metric(
            "Chunks",
            collection.count(),
        )

    st.divider()

    if indexed_documents:
        st.subheader("Indexed documents")

        for document in indexed_documents:
            st.write(
                f"**{document['source']}**"
            )
            st.caption(
                f"{document['chunk_count']} stored chunk(s)"
            )

        document_names = [
            document["source"]
            for document in indexed_documents
        ]

        selected_document = st.selectbox(
            "Select a document",
            options=document_names,
        )

        if st.button(
            "Delete selected document",
            key="delete_selected_document",
        ):
            deleted_count = delete_indexed_document(
                collection=collection,
                source=selected_document,
            )

            # Remove answers that may refer to deleted evidence.
            st.session_state.chat_history = []

            st.session_state.sidebar_message = (
                f"Deleted {selected_document} "
                f"and {deleted_count} chunk(s)."
            )

            st.rerun()

    else:
        st.info(
            "No documents are currently indexed."
        )

    st.divider()

    if st.button(
        "Clear chat history",
        key="clear_chat_history_sidebar",
    ):
        st.session_state.chat_history = []

        st.session_state.sidebar_message = (
            "Chat history cleared."
        )

        st.rerun()

    confirm_clear = st.checkbox(
        "I understand that all indexed documents "
        "will be removed."
    )

    if st.button(
        "Clear knowledge base",
        type="primary",
        disabled=not confirm_clear,
        key="clear_knowledge_base",
    ):
        deleted_count = clear_knowledge_base(
            collection
        )

        st.session_state.chat_history = []

        st.session_state.sidebar_message = (
            f"Knowledge base cleared. "
            f"{deleted_count} chunk(s) were removed."
        )

        st.rerun()

# ---------------------------------------------------------
# Document upload
# ---------------------------------------------------------

st.subheader("1. Upload a document")

uploaded_file = st.file_uploader(
    "Select a TXT or PDF file",
    type=["txt", "pdf"],
)

if uploaded_file is not None:
    st.write(
        f"Selected file: **{uploaded_file.name}**"
    )

    if st.button(
        "Process document",
        type="primary",
    ):
        try:
            with st.spinner(
                "Extracting text and creating embeddings..."
            ):
                saved_file_path = save_uploaded_file(
                    uploaded_file
                )

                document_text = load_document(
                    saved_file_path
                )

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
            st.error(
                f"Document processing failed: {error}"
            )


st.divider()


# ---------------------------------------------------------
# Agentic question answering
# ---------------------------------------------------------

if st.session_state.chat_history:
    st.subheader("Conversation history")

    for chat_item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat_item["question"])

        with st.chat_message("assistant"):
            st.markdown(chat_item["answer"])

            if chat_item["verified"]:
                st.caption("Answer passed evidence verification.")
            else:
                st.caption("Answer was not fully verified.")

    st.divider()

st.subheader("2. Ask a question")

question = st.text_input(
    "Ask a question about the uploaded documents",
    placeholder=(
        "Example: How should employees "
        "protect their company accounts?"
    ),
)

number_of_results = st.slider(
    "Number of evidence chunks",
    min_value=1,
    max_value=5,
    value=3,
)

if st.button(
    "Run Agentic RAG",
    type="primary",
):
    if not question.strip():
        st.warning(
            "Enter a question before running the agent."
        )

    else:
        try:
            with st.spinner(
                "Retrieving, generating, and verifying the answer..."
            ):
                result = agent_graph.invoke(
                    {
                        "question": question,
                        "number_of_results": number_of_results,
                    }
                )

            final_answer = result["final_answer"]
            evidence_chunks = result.get(
                "evidence_chunks",
                [],
            )

            st.session_state.chat_history.append(
    {
        "question": question,
        "answer": final_answer,
        "verified": result.get(
            "verification_passed",
            False,
        ),
    }
)

            st.subheader("Final answer")
            st.markdown(final_answer)

            if result.get("verification_passed", False):
                st.success(
                    "The answer passed evidence verification."
                )

            elif evidence_chunks:
                st.warning(
                    "The generated answer could not be fully "
                    "verified and was rejected."
                )

            correction_attempts = result.get(
                "correction_attempts",
                0,
            )

            if correction_attempts > 0:
                st.info(
                    f"The agent performed "
                    f"{correction_attempts} correction attempt."
                )

            verification_feedback = result.get(
                "verification_feedback",
                "",
            )

            if verification_feedback:
                with st.expander(
                    "Verification details"
                ):
                    st.write(
                        verification_feedback
                    )

            if evidence_chunks:
                st.subheader("Supporting evidence")

                for position, evidence in enumerate(
                    evidence_chunks,
                    start=1,
                ):
                    source = evidence["source"]
                    chunk_number = (
                        evidence["chunk_index"] + 1
                    )
                    similarity = evidence["similarity"]

                    with st.expander(
                        f"Source {position}: {source} "
                        f"— chunk {chunk_number}"
                    ):
                        st.write(
                            evidence["text"]
                        )

                        st.write(
                            f"**Similarity score:** "
                            f"{similarity:.3f}"
                        )

        except Exception as error:
            st.error(
                f"Agent workflow failed: {error}"
            )


st.divider()

st.caption(
    f"Stored chunks currently available: "
    f"{collection.count()}"
)