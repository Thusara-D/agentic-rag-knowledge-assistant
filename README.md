# Agentic RAG Knowledge Assistant

A beginner-friendly portfolio project that answers questions from private
documents and checks its own answer against retrieved evidence.

## Planned architecture

`Documents -> Text chunks -> Embeddings -> Chroma -> Retrieval -> LLM -> Verification -> Cited answer`

## Technology

- Python
- LangGraph
- LangChain
- Chroma
- OpenAI
- Streamlit
- Pytest

## Current status

Milestone 1: project foundation and basic Streamlit application.

## Local setup on Windows

Open the project folder in Visual Studio or Visual Studio Code, and then open
its terminal.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this command once in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

## Run tests

```powershell
pytest
```

## Environment variables

Copy `.env.example` to `.env` and insert the real API key only when the
OpenAI integration is added.

Never commit `.env`.

## Daily Git workflow

Before working:

```powershell
git pull origin main
```

After completing one meaningful change:

```powershell
git status
git add .
git commit -m "describe the completed change"
git push origin main
```

## Portfolio goals

The finished project should demonstrate:

- document ingestion and chunking;
- vector embeddings and semantic retrieval;
- grounded LLM responses;
- an agentic verification loop;
- citations and traceable evidence;
- testing, documentation, and source control.

## Important limitation

The assistant must state when the supplied documents do not contain enough
evidence to answer a question.
