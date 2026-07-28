import os

from dotenv import load_dotenv


load_dotenv()

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def get_gemini_api_key() -> str:
    """Return the Gemini API key without displaying it."""

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. "
            "Add your Gemini API key to the .env file."
        )

    return api_key


def get_gemini_model() -> str:
    """Return the configured Gemini model name."""

    model_name = os.getenv(
        "GEMINI_MODEL",
        DEFAULT_GEMINI_MODEL,
    ).strip()

    if not model_name:
        return DEFAULT_GEMINI_MODEL

    return model_name