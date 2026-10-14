import os
from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class Settings:
    GOOGLE_GENERATIVE_AI_API_KEY: str = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
        or ""
    )
    ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
    # Shared with focusly-workflows (X-Internal-Token). When set, only callers
    # that send it can use the AI endpoints (and spend the API keys).
    # Gemini models by tier (see app/services/ai/models.py).
    GEMINI_LIGHT_MODEL = os.environ.get(
        "AI_GEMINI_LIGHT_MODEL", "gemini-3.5-flash-lite"
    ).strip()
    GEMINI_STRONG_MODEL = os.environ.get(
        "AI_GEMINI_STRONG_MODEL", "gemini-2.5-flash"
    ).strip()
    INTERNAL_TOKEN: str = os.environ.get("FOCUSLY_AI_INTERNAL_TOKEN", "").strip()
    # Caps on what one reply can cost. Long documents still fit; runaway
    # output and unbounded "thinking" don't.
    MAX_OUTPUT_TOKENS: int = _int("AI_MAX_OUTPUT_TOKENS", 16384)
    THINKING_BUDGET_FLASH: int = _int("AI_THINKING_BUDGET_FLASH", 1024)
    THINKING_BUDGET_PRO: int = _int("AI_THINKING_BUDGET_PRO", 2048)
    # Planner/JSON endpoints: structured answers, little reasoning needed.
    THINKING_BUDGET_STRUCTURED: int = _int("AI_THINKING_BUDGET_STRUCTURED", 512)
    REQUEST_TIMEOUT_SECONDS: int = _int("AI_REQUEST_TIMEOUT_SECONDS", 60)


settings = Settings()
