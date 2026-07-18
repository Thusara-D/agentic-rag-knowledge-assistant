import os

from dotenv import load_dotenv

load_dotenv()


def get_openai_api_key() -> str:
    """Return the OpenAI API key without printing or exposing it."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Copy .env.example to .env "
            "and add your key."
        )

    return api_key
