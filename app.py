import streamlit as st

st.set_page_config(
    page_title="Agentic RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("Agentic RAG Knowledge Assistant")
st.caption("A portfolio project for document-grounded, self-checking AI answers.")

st.success("Milestone 1 is working: the project environment and user interface are ready.")

st.subheader("Planned workflow")
st.write("1. Upload documents")
st.write("2. Split documents into chunks")
st.write("3. Store embeddings in Chroma")
st.write("4. Retrieve relevant evidence")
st.write("5. Generate an answer")
st.write("6. Verify the answer against the evidence")
st.write("7. Display citations")

st.info("The RAG functions will be implemented one milestone at a time.")
