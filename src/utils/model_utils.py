"""Shared model classification utilities."""

_LIGHT_MODEL_MARKERS = ("haiku", "mini", "flash", "3.5-turbo")


def is_light_model(model: str) -> bool:
    """Check if a model is a light/cheap model.

    Matches: Haiku, GPT-4o-mini, GPT-3.5-turbo, Gemini Flash, etc.
    """
    lower = model.lower()
    return any(marker in lower for marker in _LIGHT_MODEL_MARKERS)
