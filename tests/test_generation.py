from unittest.mock import MagicMock, patch

import pytest

from src.generation import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    build_evidence_context,
    generate_grounded_answer,
)


def test_build_evidence_context():
    """Evidence should be formatted with source labels."""

    evidence = [
        {
            "text": "Employees must use multi-factor authentication.",
            "source": "security_policy.txt",
            "chunk_index": 0,
        }
    ]

    context = build_evidence_context(evidence)

    assert "[Source 1: security_policy.txt, chunk 1]" in context
    assert "Employees must use multi-factor authentication." in context


def test_build_evidence_context_rejects_empty_list():
    """At least one evidence chunk is required."""

    with pytest.raises(
        ValueError,
        match="At least one evidence chunk is required",
    ):
        build_evidence_context([])


def test_generate_answer_rejects_empty_question():
    """An empty question should raise an error."""

    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        generate_grounded_answer(
            question="   ",
            evidence_chunks=[
                {
                    "text": "Example evidence",
                    "source": "example.txt",
                    "chunk_index": 0,
                }
            ],
        )


def test_generate_answer_returns_insufficient_message_without_evidence():
    """No evidence should return the standard insufficient-evidence message."""

    answer = generate_grounded_answer(
        question="What is the policy?",
        evidence_chunks=[],
    )

    assert answer == INSUFFICIENT_EVIDENCE_MESSAGE


@patch("src.generation.genai.Client")
def test_generate_grounded_answer_uses_gemini(
    mock_client_class,
):
    """Gemini should receive the question and evidence."""

    mock_response = MagicMock()
    mock_response.text = (
        "Employees must use multi-factor authentication. [Source 1]"
    )

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    mock_client_class.return_value = mock_client

    evidence = [
        {
            "text": "Employees must use multi-factor authentication.",
            "source": "security_policy.txt",
            "chunk_index": 0,
        }
    ]

    answer = generate_grounded_answer(
        question="How should employees secure their accounts?",
        evidence_chunks=evidence,
    )

    assert "multi-factor authentication" in answer
    assert "[Source 1]" in answer

    mock_client.models.generate_content.assert_called_once()


@patch("src.generation.genai.Client")
def test_generate_answer_rejects_empty_gemini_response(
    mock_client_class,
):
    """An empty Gemini response should raise an error."""

    mock_response = MagicMock()
    mock_response.text = ""

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    mock_client_class.return_value = mock_client

    with pytest.raises(
        ValueError,
        match="Gemini returned an empty response",
    ):
        generate_grounded_answer(
            question="What is the security policy?",
            evidence_chunks=[
                {
                    "text": "Employees must use MFA.",
                    "source": "policy.txt",
                    "chunk_index": 0,
                }
            ],
        )