"""The LangGraph agent workflow will be implemented in Milestone 5."""

from typing import Any, Literal, TypedDict

from google import genai
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.config import (
    get_gemini_api_key,
    get_gemini_model,
)
from src.generation import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    build_evidence_context,
    generate_grounded_answer,
)
from src.retrieval import search_similar_chunks


MAX_CORRECTION_ATTEMPTS = 1


class AgentState(TypedDict, total=False):
    """Information shared between all LangGraph nodes."""

    question: str
    number_of_results: int
    evidence_chunks: list[dict[str, Any]]
    draft_answer: str
    verification_passed: bool
    verification_feedback: str
    correction_attempts: int
    final_answer: str


class VerificationDecision(BaseModel):
    """Structured decision returned by the verifier."""

    supported: bool = Field(
        description=(
            "True only when every factual claim in the answer "
            "is supported by the supplied evidence."
        )
    )

    feedback: str = Field(
        description=(
            "A brief explanation of unsupported claims or "
            "missing evidence."
        )
    )


def verify_grounded_answer(
    question: str,
    answer: str,
    evidence_chunks: list[dict[str, Any]],
) -> VerificationDecision:
    """Check whether an answer is fully supported by evidence."""

    if not answer.strip():
        raise ValueError("Answer cannot be empty.")

    if not evidence_chunks:
        return VerificationDecision(
            supported=False,
            feedback="No evidence was available.",
        )

    evidence_context = build_evidence_context(
        evidence_chunks
    )

    prompt = f"""
You are a strict factual verification system.

Check whether the draft answer is fully supported by the evidence.

Verification rules:
1. Every factual claim must be supported by the evidence.
2. Do not use outside knowledge.
3. Citations must refer to the correct source labels.
4. Mark the answer unsupported if it adds, assumes, or invents facts.
5. Provide short and specific feedback.

User question:
{question}

Draft answer:
{answer}

Evidence:
{evidence_context}
""".strip()

    client = genai.Client(
        api_key=get_gemini_api_key()
    )

    response = client.models.generate_content(
        model=get_gemini_model(),
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": (
                VerificationDecision.model_json_schema()
            ),
        },
    )

    response_text = (response.text or "").strip()

    if not response_text:
        raise ValueError(
            "Gemini verifier returned an empty response."
        )

    return VerificationDecision.model_validate_json(
        response_text
    )


def correct_grounded_answer(
    question: str,
    draft_answer: str,
    verification_feedback: str,
    evidence_chunks: list[dict[str, Any]],
) -> str:
    """Correct an unsupported answer using verifier feedback."""

    evidence_context = build_evidence_context(
        evidence_chunks
    )

    prompt = f"""
You are correcting a document-grounded answer.

Rewrite the draft answer so every factual claim is supported
by the supplied evidence.

Rules:
1. Use only the supplied evidence.
2. Remove unsupported or invented claims.
3. Include source labels such as [Source 1].
4. Follow the verifier feedback.
5. If the evidence is insufficient, respond exactly with:
   "{INSUFFICIENT_EVIDENCE_MESSAGE}"

User question:
{question}

Previous draft:
{draft_answer}

Verifier feedback:
{verification_feedback}

Evidence:
{evidence_context}

Corrected answer:
""".strip()

    client = genai.Client(
        api_key=get_gemini_api_key()
    )

    response = client.models.generate_content(
        model=get_gemini_model(),
        contents=prompt,
    )

    corrected_answer = (response.text or "").strip()

    if not corrected_answer:
        raise ValueError(
            "Gemini returned an empty corrected answer."
        )

    return corrected_answer


def build_agent_graph(collection: Any) -> Any:
    """Build and compile the complete Agentic RAG workflow."""

    def retrieve_node(
        state: AgentState,
    ) -> dict[str, Any]:
        """Retrieve relevant evidence from Chroma."""

        evidence = search_similar_chunks(
            collection=collection,
            question=state["question"],
            number_of_results=state.get(
                "number_of_results",
                3,
            ),
        )

        return {
            "evidence_chunks": evidence,
            "correction_attempts": 0,
            "verification_passed": False,
            "verification_feedback": "",
        }

    def generate_node(
        state: AgentState,
    ) -> dict[str, str]:
        """Generate the first grounded answer."""

        answer = generate_grounded_answer(
            question=state["question"],
            evidence_chunks=state["evidence_chunks"],
        )

        return {
            "draft_answer": answer,
        }

    def verify_node(
        state: AgentState,
    ) -> dict[str, Any]:
        """Verify the generated answer."""

        decision = verify_grounded_answer(
            question=state["question"],
            answer=state["draft_answer"],
            evidence_chunks=state["evidence_chunks"],
        )

        return {
            "verification_passed": decision.supported,
            "verification_feedback": decision.feedback,
        }

    def correct_node(
        state: AgentState,
    ) -> dict[str, Any]:
        """Correct an answer that failed verification."""

        corrected_answer = correct_grounded_answer(
            question=state["question"],
            draft_answer=state["draft_answer"],
            verification_feedback=state[
                "verification_feedback"
            ],
            evidence_chunks=state["evidence_chunks"],
        )

        return {
            "draft_answer": corrected_answer,
            "correction_attempts": (
                state.get("correction_attempts", 0) + 1
            ),
        }

    def finalize_node(
        state: AgentState,
    ) -> dict[str, str]:
        """Return the verified answer."""

        final_answer = state.get(
            "draft_answer",
            INSUFFICIENT_EVIDENCE_MESSAGE,
        )

        return {
            "final_answer": final_answer,
        }

    def reject_node(
        state: AgentState,
    ) -> dict[str, str]:
        """Reject an answer that remains unsupported."""

        return {
            "final_answer": (
                INSUFFICIENT_EVIDENCE_MESSAGE
            ),
        }

    def route_after_retrieval(
        state: AgentState,
    ) -> Literal["generate", "finalize"]:
        """Skip generation when no evidence was found."""

        if state.get("evidence_chunks"):
            return "generate"

        return "finalize"

    def route_after_verification(
        state: AgentState,
    ) -> Literal["finalize", "correct", "reject"]:
        """Choose whether to finish, correct, or reject."""

        if state.get("verification_passed", False):
            return "finalize"

        if (
            state.get("correction_attempts", 0)
            < MAX_CORRECTION_ATTEMPTS
        ):
            return "correct"

        return "reject"

    builder = StateGraph(AgentState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("verify", verify_node)
    builder.add_node("correct", correct_node)
    builder.add_node("finalize", finalize_node)
    builder.add_node("reject", reject_node)

    builder.add_edge(
        START,
        "retrieve",
    )

    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "generate": "generate",
            "finalize": "finalize",
        },
    )

    builder.add_edge(
        "generate",
        "verify",
    )

    builder.add_conditional_edges(
        "verify",
        route_after_verification,
        {
            "finalize": "finalize",
            "correct": "correct",
            "reject": "reject",
        },
    )

    builder.add_edge(
        "correct",
        "verify",
    )

    builder.add_edge(
        "finalize",
        END,
    )

    builder.add_edge(
        "reject",
        END,
    )

    return builder.compile()