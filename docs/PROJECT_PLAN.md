# Project Plan

## Milestone 1 — Repository foundation
- Create the Python project.
- Create a virtual environment.
- Add `.gitignore` and `.env.example`.
- Run a basic Streamlit interface.
- Add one automated smoke test.

Suggested commit:

`chore: initialize project structure and Streamlit app`

## Milestone 2 — Document ingestion
- Upload TXT and PDF files.
- Extract text.
- Split text into chunks.
- Preserve source metadata.

Suggested commit:

`feat: add document loading and text chunking`

## Milestone 3 — Vector storage and retrieval
- Generate embeddings.
- Store chunks in Chroma.
- Retrieve relevant chunks for a question.

Suggested commit:

`feat: add Chroma vector store and semantic retrieval`

## Milestone 4 — Grounded RAG answering
- Send retrieved context to the language model.
- Require answers to use only supplied evidence.
- Return source references.

Suggested commit:

`feat: add evidence-grounded question answering`

## Milestone 5 — Agentic workflow
- Add LangGraph state.
- Create retrieve, answer, verify, and finalize nodes.
- Route weak answers back for correction.

Suggested commit:

`feat: implement LangGraph verification workflow`

## Milestone 6 — User interface
- Add document upload.
- Add chat history.
- Show retrieved evidence and citations.
- Add useful error messages.

Suggested commit:

`feat: complete interactive Streamlit RAG interface`

## Milestone 7 — Quality and portfolio presentation
- Add unit tests.
- Add architecture diagram.
- Add screenshots and demonstration questions.
- Add limitations and future improvements.
- Add GitHub Actions.

Suggested commit:

`docs: complete portfolio documentation and demo`
