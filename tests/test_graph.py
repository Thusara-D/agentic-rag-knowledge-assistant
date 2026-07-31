from unittest.mock import MagicMock, patch

from src.generation import INSUFFICIENT_EVIDENCE_MESSAGE
from src.graph import (
    VerificationDecision,
    build_agent_graph,
)


SAMPLE_EVIDENCE = [
    {
        "text": "Employees must use multi-factor authentication.",
        "source": "security_policy.txt",
        "chunk_index": 0,
        "distance": 0.1,
        "similarity": 0.9,
    }
]


def test_graph_returns_verified_answer():
    """A supported first draft should become the final answer."""

    fake_collection = MagicMock()

    with (
        patch(
            "src.graph.search_similar_chunks",
            return_value=SAMPLE_EVIDENCE,
        ),
        patch(
            "src.graph.generate_grounded_answer",
            return_value=(
                "Employees must use multi-factor "
                "authentication. [Source 1]"
            ),
        ),
        patch(
            "src.graph.verify_grounded_answer",
            return_value=VerificationDecision(
                supported=True,
                feedback="The answer is fully supported.",
            ),
        ),
    ):
        graph = build_agent_graph(fake_collection)

        result = graph.invoke(
            {
                "question": (
                    "How should employees protect "
                    "their accounts?"
                ),
                "number_of_results": 3,
            }
        )

    assert result["verification_passed"] is True
    assert "multi-factor authentication" in result["final_answer"]
    assert result["correction_attempts"] == 0


def test_graph_corrects_unsupported_answer():
    """An unsupported draft should be corrected and verified again."""

    fake_collection = MagicMock()

    verification_results = [
        VerificationDecision(
            supported=False,
            feedback="The draft contains an unsupported claim.",
        ),
        VerificationDecision(
            supported=True,
            feedback="The corrected answer is supported.",
        ),
    ]

    with (
        patch(
            "src.graph.search_similar_chunks",
            return_value=SAMPLE_EVIDENCE,
        ),
        patch(
            "src.graph.generate_grounded_answer",
            return_value=(
                "Employees must use MFA and change "
                "passwords every week. [Source 1]"
            ),
        ),
        patch(
            "src.graph.correct_grounded_answer",
            return_value=(
                "Employees must use multi-factor "
                "authentication. [Source 1]"
            ),
        ),
        patch(
            "src.graph.verify_grounded_answer",
            side_effect=verification_results,
        ) as mock_verifier,
    ):
        graph = build_agent_graph(fake_collection)

        result = graph.invoke(
            {
                "question": (
                    "How should employees protect "
                    "their accounts?"
                ),
                "number_of_results": 3,
            }
        )

    assert result["verification_passed"] is True
    assert result["correction_attempts"] == 1
    assert "change passwords every week" not in result["final_answer"]
    assert mock_verifier.call_count == 2


def test_graph_rejects_answer_that_remains_unsupported():
    """The graph should reject an answer that fails twice."""

    fake_collection = MagicMock()

    unsupported_decision = VerificationDecision(
        supported=False,
        feedback="The answer is not supported by the evidence.",
    )

    with (
        patch(
            "src.graph.search_similar_chunks",
            return_value=SAMPLE_EVIDENCE,
        ),
        patch(
            "src.graph.generate_grounded_answer",
            return_value="Unsupported first answer.",
        ),
        patch(
            "src.graph.correct_grounded_answer",
            return_value="Unsupported corrected answer.",
        ),
        patch(
            "src.graph.verify_grounded_answer",
            return_value=unsupported_decision,
        ) as mock_verifier,
    ):
        graph = build_agent_graph(fake_collection)

        result = graph.invoke(
            {
                "question": "What is the company policy?",
                "number_of_results": 3,
            }
        )

    assert result["verification_passed"] is False
    assert result["correction_attempts"] == 1
    assert result["final_answer"] == INSUFFICIENT_EVIDENCE_MESSAGE
    assert mock_verifier.call_count == 2


def test_graph_handles_empty_knowledge_base():
    """No retrieved evidence should skip Gemini generation."""

    fake_collection = MagicMock()

    with (
        patch(
            "src.graph.search_similar_chunks",
            return_value=[],
        ),
        patch(
            "src.graph.generate_grounded_answer",
        ) as mock_generator,
        patch(
            "src.graph.verify_grounded_answer",
        ) as mock_verifier,
    ):
        graph = build_agent_graph(fake_collection)

        result = graph.invoke(
            {
                "question": "What is the company policy?",
                "number_of_results": 3,
            }
        )

    assert result["evidence_chunks"] == []
    assert result["final_answer"] == INSUFFICIENT_EVIDENCE_MESSAGE
    mock_generator.assert_not_called()
    mock_verifier.assert_not_called()