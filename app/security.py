import hmac
import logging

from fastapi import Header, HTTPException

from app.config import settings

logger = logging.getLogger(__name__)

if not settings.INTERNAL_TOKEN:
    logger.warning(
        "FOCUSLY_AI_INTERNAL_TOKEN is not set: focusly-ai accepts requests from "
        "anyone who can reach it. Set it (same value in focusly-workflows)."
    )


async def require_internal_token(
    x_internal_token: str | None = Header(default=None),
) -> None:
    """Only focusly-workflows may call the AI endpoints: it checks the user's
    plan and limits before spending tokens on them."""
    expected = settings.INTERNAL_TOKEN
    if not expected:
        return
    if not x_internal_token or not hmac.compare_digest(x_internal_token, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")
