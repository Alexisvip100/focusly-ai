from typing import Any

from app.config import settings

# The Gemini models Focusly uses, by tier. Clients name a tier ("…flash-lite"
# or "…flash"); the exact ids live in the settings, so a model Google retires
# is swapped there.


def light_model() -> str:
    return settings.GEMINI_LIGHT_MODEL


def strong_model() -> str:
    return settings.GEMINI_STRONG_MODEL


def resolve_gemini_model(model: str | None) -> str:
    """The light model unless the strong one is asked for. Retired names
    (1.5, 2.0), Claude names and unknown ones get the light model."""
    name = (model or "").lower()
    if name in (settings.GEMINI_LIGHT_MODEL, settings.GEMINI_STRONG_MODEL):
        return name
    if "gemini" in name and "lite" not in name and ("flash" in name or "pro" in name):
        return settings.GEMINI_STRONG_MODEL
    return settings.GEMINI_LIGHT_MODEL


def is_light(model: str) -> bool:
    return model == settings.GEMINI_LIGHT_MODEL or "lite" in model


def thinking_config(model: str, *, structured: bool = False) -> dict[str, Any]:
    """How much the model may think (billed as output). Gemini 2.x takes a
    token budget; Gemini 3 and later take a level."""
    if model.startswith("gemini-2"):
        if is_light(model):
            budget = 0
        elif "pro" in model:
            # 2.5 Pro can't turn thinking off.
            budget = max(128, settings.THINKING_BUDGET_PRO)
        elif structured:
            budget = settings.THINKING_BUDGET_STRUCTURED
        else:
            budget = settings.THINKING_BUDGET_FLASH
        return {"thinkingBudget": budget}
    return {"thinkingLevel": "minimal" if is_light(model) else "low"}


def sdk_thinking_config(model: str, *, structured: bool = False) -> dict[str, Any]:
    """thinking_config with the google-genai SDK's field names."""
    config = thinking_config(model, structured=structured)
    if "thinkingBudget" in config:
        return {"thinking_budget": config["thinkingBudget"]}
    return {"thinking_level": config["thinkingLevel"]}
