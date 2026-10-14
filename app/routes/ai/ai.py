from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any
import httpx
import json

from app.config import settings
from app.services.ai.models import (
    is_light,
    resolve_gemini_model,
    thinking_config,
)
from app.services.ai.prompts import (
    GOLDEN_HOURS_SYSTEM_INSTRUCTION,
    GOLDEN_HOURS_USER_PROMPT,
)

router = APIRouter(tags=["ai"])

# The reply text ends here when usage was requested: what follows the marker
# (ASCII record separator, never in model text) is the usage JSON.
USAGE_MARKER = "\x1e"

CLAUDE_MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
}

# ─── Schemas ──────────────────────────────────────────────────────────────────


class MessageSchema(BaseModel):
    role: str
    content: str


class ChatRequestSchema(BaseModel):
    messages: list[MessageSchema]
    system_context: str
    model: str | None = None
    # Ask for the token usage at the end of the stream, after USAGE_MARKER.
    include_usage: bool = False
    # system_context[:cache_prefix_chars] repeats between messages (the
    # instructions, the user's tasks…); the rest changes (time, memories).
    cache_prefix_chars: int | None = None


class AnalyzePatternsRequest(BaseModel):
    user_name: str
    hour_buckets: dict[str, Any]
    task_stats: dict[str, Any]
    session_stats: dict[str, Any]
    top_productive_hours: list[int]
    work_style_hint: str


# ─── Models and generation settings ──────────────────────────────────────────


def resolve_claude_model(model: str | None) -> str:
    """A current Claude model for a requested name or Gemini tier."""
    name = (model or "").lower()
    if "opus" in name or name.endswith("-pro"):
        return CLAUDE_MODELS["opus"]
    if "haiku" in name or not name or is_light(name):
        return CLAUDE_MODELS["haiku"]
    return CLAUDE_MODELS["sonnet"]


def gemini_generation_config(model: str, *, structured: bool = False) -> dict:
    """Output cap plus a thinking cap: thinking is billed as output, and the
    flash models think by default."""
    return {
        "maxOutputTokens": settings.MAX_OUTPUT_TOKENS,
        "thinkingConfig": thinking_config(model, structured=structured),
    }


def usage_trailer(usage: dict) -> str:
    return USAGE_MARKER + json.dumps(usage)


# ─── Gemini Stream Parser ─────────────────────────────────────────────────────


class GeminiStreamParser:
    def __init__(self):
        self.buffer = ""
        # The last usageMetadata seen (Gemini sends it with the final chunk).
        self.usage: dict[str, Any] | None = None

    def feed(self, chunk: str):
        self.buffer += chunk
        self.buffer = self.buffer.lstrip("[\r\n, ")

        while self.buffer:
            if not self.buffer.startswith("{"):
                self.buffer = self.buffer.lstrip("\r\n, ]")
                if not self.buffer:
                    break
                if not self.buffer.startswith("{"):
                    self.buffer = ""
                    break

            brace_count = 0
            in_string = False
            escape = False
            end_idx = -1

            for idx, char in enumerate(self.buffer):
                if char == '"' and not escape:
                    in_string = not in_string
                elif char == "\\" and in_string:
                    escape = not escape
                else:
                    escape = False

                if not in_string:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = idx
                            break

            if end_idx != -1:
                obj_str = self.buffer[: end_idx + 1]
                self.buffer = self.buffer[end_idx + 1 :].lstrip("\r\n, ")
                try:
                    obj = json.loads(obj_str)
                    if obj.get("usageMetadata"):
                        self.usage = obj["usageMetadata"]
                    candidates = obj.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        # Thought summaries (if any) are not part of the reply.
                        for part in parts:
                            text = part.get("text", "")
                            if text and not part.get("thought"):
                                yield text
                except Exception:
                    pass
            else:
                break


# ─── Streaming Assistant ──────────────────────────────────────────────────────

TOO_MANY_REQUESTS = "Lumina ha alcanzado el límite de consultas permitidas por hoy. ⏳ Por favor, dale un respiro e intenta de nuevo en unos minutos. ¡Volveré pronto para ayudarte! 🌟"
OVERLOADED = "¡Hola! Lumina está recibiendo muchísimas consultas en este momento y está un poco ocupada. 📭 Por favor, intenta de nuevo en unos segundos. ¡Estaré lista para ti de inmediato! ☕"
TEMPORARY_ERROR = "Lumina ha experimentado un problema técnico temporal al procesar tu solicitud. 🛠️ Por favor, intenta de nuevo en un momento. ¡Gracias por tu paciencia! ✨"
STREAM_BROKEN = "\nLumina ha experimentado un problema técnico temporal al procesar tu solicitud. 🛠️ Por favor, intenta de nuevo en un momento."


def friendly_error(status_code: int) -> str:
    if status_code == 429:
        return TOO_MANY_REQUESTS
    if status_code == 503:
        return OVERLOADED
    return TEMPORARY_ERROR


# What stream_claude yields when it fails before answering.
CLAUDE_FAILURES = {TOO_MANY_REQUESTS, OVERLOADED, TEMPORARY_ERROR}


async def stream_gemini(payload: dict, model: str, usage: dict | None = None):
    """Streams Gemini's reply text. When `usage` is given, it's filled with
    the token counts once the stream ends."""
    api_key = settings.GOOGLE_GENERATIVE_AI_API_KEY
    if not api_key:
        yield "Error: GEMINI_API_KEY is not configured in focusly-ai settings."
        return

    model = resolve_gemini_model(model)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?key={api_key}"
    parser = GeminiStreamParser()

    async with httpx.AsyncClient() as client:
        try:
            async with client.stream(
                "POST", url, json=payload, timeout=settings.REQUEST_TIMEOUT_SECONDS
            ) as r:
                if r.status_code != 200:
                    error_body = await r.aread()  # consume connection
                    detail = (
                        error_body.decode("utf-8", errors="ignore")[:500]
                        if isinstance(error_body, bytes)
                        else ""
                    )
                    print(
                        f"[stream_gemini] {model} answered {r.status_code}: {detail}",
                        flush=True,
                    )
                    yield friendly_error(r.status_code)
                    return

                async for chunk in r.aiter_text():
                    for text in parser.feed(chunk):
                        yield text
        except Exception:
            yield STREAM_BROKEN
            return

    if usage is not None and parser.usage:
        u = parser.usage
        usage.update(
            provider="gemini",
            model=model,
            input=u.get("promptTokenCount", 0),
            cached=u.get("cachedContentTokenCount", 0),
            output=u.get("candidatesTokenCount", 0),
            thoughts=u.get("thoughtsTokenCount", 0),
        )


async def stream_claude(payload: dict, usage: dict | None = None):
    """Streams Claude's reply text. A failure before any text yields one of
    CLAUDE_FAILURES (the caller may fall back to Gemini on it)."""
    api_key = settings.ANTHROPIC_API_KEY
    if not api_key:
        yield "Error: ANTHROPIC_API_KEY is not configured."
        return

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    counts: dict[str, int] = {}
    streamed_any = False

    async with httpx.AsyncClient() as client:
        try:
            async with client.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
                timeout=settings.REQUEST_TIMEOUT_SECONDS,
            ) as r:
                if r.status_code != 200:
                    error_body = await r.aread()
                    print(
                        f"[stream_claude] non-200 from Anthropic: {r.status_code} - {(error_body or b'').decode('utf-8', errors='ignore')}",
                        flush=True,
                    )
                    yield friendly_error(r.status_code)
                    return

                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    kind = data.get("type")
                    if kind == "content_block_delta":
                        text = data.get("delta", {}).get("text", "")
                        if text:
                            streamed_any = True
                            yield text
                    elif kind == "message_start":
                        u = data.get("message", {}).get("usage", {}) or {}
                        counts["input"] = u.get("input_tokens", 0)
                        counts["cached"] = u.get("cache_read_input_tokens", 0)
                        counts["cache_write"] = u.get("cache_creation_input_tokens", 0)
                    elif kind == "message_delta":
                        u = data.get("usage", {}) or {}
                        counts["output"] = u.get("output_tokens", 0)
                    elif kind == "error":
                        print(f"[stream_claude] API error event: {data}", flush=True)
        except Exception as e:
            print(f"[stream_claude] exception: {e!r}", flush=True)
            yield STREAM_BROKEN if streamed_any else TEMPORARY_ERROR
            return

    if usage is not None and counts:
        usage.update(
            provider="claude",
            model=payload.get("model"),
            input=counts.get("input", 0),
            cached=counts.get("cached", 0),
            cache_write=counts.get("cache_write", 0),
            output=counts.get("output", 0),
            thoughts=0,
        )


def claude_system_blocks(body: ChatRequestSchema) -> list[dict]:
    """The system prompt with a cache breakpoint after its stable part: later
    turns read that part at a fraction of the price."""
    text = body.system_context
    cut = body.cache_prefix_chars
    if not cut or cut >= len(text):
        return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
    return [
        {
            "type": "text",
            "text": text[:cut],
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": text[cut:]},
    ]


def build_claude_payload(body: ChatRequestSchema) -> dict:
    return {
        "model": resolve_claude_model(body.model),
        "max_tokens": settings.MAX_OUTPUT_TOKENS,
        "system": claude_system_blocks(body),
        "messages": [
            {
                "role": "user" if m.role == "user" else "assistant",
                "content": m.content,
            }
            for m in body.messages
        ],
        "stream": True,
    }


def build_gemini_payload(body: ChatRequestSchema, model: str) -> dict:
    return {
        "contents": [
            {
                "role": "user" if m.role == "user" else "model",
                "parts": [{"text": m.content}],
            }
            for m in body.messages
        ],
        "systemInstruction": {"parts": [{"text": body.system_context}]},
        "generationConfig": gemini_generation_config(model),
    }


# ─── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/chat")
async def chat_endpoint(body: ChatRequestSchema):
    if not body.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty")

    async def stream_chat():
        usage: dict[str, Any] | None = {} if body.include_usage else None
        # Only pass `usage` when asked: the streamers can be swapped in tests.
        usage_kwargs = {"usage": usage} if usage is not None else {}

        # 1. Claude, when configured; Gemini if it fails before answering.
        if settings.ANTHROPIC_API_KEY:
            failed_with = None
            first = True
            async for text in stream_claude(build_claude_payload(body), **usage_kwargs):
                if first and text in CLAUDE_FAILURES:
                    failed_with = text
                    break
                first = False
                yield text
            if failed_with is None:
                if usage:
                    yield usage_trailer(usage)
                return
            if not settings.GOOGLE_GENERATIVE_AI_API_KEY:
                yield failed_with
                return
            print("[chat] Claude failed, falling back to Gemini", flush=True)

        # 2. Gemini
        if not settings.GOOGLE_GENERATIVE_AI_API_KEY:
            yield "Error: No AI provider (Claude or Gemini) is properly configured in focusly-ai."
            return

        model = resolve_gemini_model(body.model)
        async for text in stream_gemini(
            build_gemini_payload(body, model), model, **usage_kwargs
        ):
            yield text
        if usage:
            yield usage_trailer(usage)

    return StreamingResponse(stream_chat(), media_type="text/plain")


@router.post("/analyze-patterns")
async def analyze_patterns_endpoint(body: AnalyzePatternsRequest):
    api_key = settings.GOOGLE_GENERATIVE_AI_API_KEY
    if not api_key:
        raise HTTPException(
            status_code=500, detail="Gemini API key is not configured in focusly-ai"
        )

    system_instruction = GOLDEN_HOURS_SYSTEM_INSTRUCTION.format(
        user_name=body.user_name
    )

    user_prompt = GOLDEN_HOURS_USER_PROMPT.format(
        user_name=body.user_name,
        hour_buckets=json.dumps(body.hour_buckets, indent=2),
        task_stats=json.dumps(body.task_stats, indent=2),
        session_stats=json.dumps(body.session_stats, indent=2),
        top_productive_hours=body.top_productive_hours,
        work_style_hint=body.work_style_hint,
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.GEMINI_STRONG_MODEL}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            **gemini_generation_config(settings.GEMINI_STRONG_MODEL, structured=True),
        },
    }

    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(url, json=payload, timeout=30.0)
            if r.status_code != 200:
                raise HTTPException(
                    status_code=502, detail=f"Gemini API returned code {r.status_code}"
                )
            res_json = r.json()
            text_response = res_json["candidates"][0]["content"]["parts"][0]["text"]

            analysis = json.loads(text_response.strip())
            return {"success": True, "data": analysis}
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error analyzing patterns in focusly-ai: {str(e)}",
            )
