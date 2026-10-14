"""
tests/test_ai_optimization.py

Gasto de tokens: modelos vigentes, topes de salida y de thinking, el uso de
tokens al final del stream, caché del system prompt de Claude, el fallback a
Gemini y el token interno entre focusly-workflows y focusly-ai.
"""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.routes.ai.ai import (
    CLAUDE_MODELS,
    TOO_MANY_REQUESTS,
    USAGE_MARKER,
    GeminiStreamParser,
    gemini_generation_config,
    resolve_claude_model,
    resolve_gemini_model,
    settings,
    stream_claude,
)
from tests.test_stream_errors import collect_stream, make_httpx_mock

CHAT = {
    "messages": [{"role": "user", "content": "Hola"}],
    "system_context": "Eres Lumina.",
}


# ─── Models ────────────────────────────────────────────────────────────────────


LIGHT = settings.GEMINI_LIGHT_MODEL
STRONG = settings.GEMINI_STRONG_MODEL


class TestModels:
    @pytest.mark.parametrize(
        "requested, expected",
        [
            (None, LIGHT),
            ("", LIGHT),
            (LIGHT, LIGHT),
            (STRONG, STRONG),
            # Clients name a tier; the exact ids are settings.
            ("gemini-2.5-flash-lite", LIGHT),
            ("gemini-3.5-flash", STRONG),
            ("gemini-2.5-pro", STRONG),
            # Foreign or unknown names go to the cheapest model.
            ("claude-3-5-sonnet", LIGHT),
            ("auto", LIGHT),
        ],
    )
    def test_resolve_gemini_model(self, requested, expected):
        assert resolve_gemini_model(requested) == expected

    @pytest.mark.parametrize(
        "requested, tier",
        [
            ("claude-3-5-haiku", "haiku"),
            (LIGHT, "haiku"),
            (None, "haiku"),
            ("claude-3-opus", "opus"),
            ("gemini-2.5-pro", "opus"),
            ("claude-3-5-sonnet", "sonnet"),
            ("gemini-2.5-flash", "sonnet"),
        ],
    )
    def test_resolve_claude_model(self, requested, tier):
        assert resolve_claude_model(requested) == CLAUDE_MODELS[tier]


# ─── Generation config ─────────────────────────────────────────────────────────


class TestGenerationConfig:
    def test_gemini_2_light_does_not_think(self):
        config = gemini_generation_config("gemini-2.5-flash-lite")
        assert config["thinkingConfig"] == {"thinkingBudget": 0}
        assert config["maxOutputTokens"] == settings.MAX_OUTPUT_TOKENS

    def test_gemini_2_flash_thinking_is_capped(self):
        config = gemini_generation_config("gemini-2.5-flash")
        assert config["thinkingConfig"] == {
            "thinkingBudget": settings.THINKING_BUDGET_FLASH
        }

    def test_structured_output_thinks_less(self):
        config = gemini_generation_config("gemini-2.5-flash", structured=True)
        assert config["thinkingConfig"] == {
            "thinkingBudget": settings.THINKING_BUDGET_STRUCTURED
        }

    def test_pro_keeps_a_minimum_budget(self, monkeypatch):
        # Gemini 2.5 Pro rejects a budget under 128 (it can't stop thinking).
        monkeypatch.setattr("app.routes.ai.ai.settings.THINKING_BUDGET_PRO", 0)
        config = gemini_generation_config("gemini-2.5-pro")
        assert config["thinkingConfig"]["thinkingBudget"] == 128

    def test_gemini_3_takes_levels(self):
        # Gemini 3+ rejects thinkingBudget: 0 ("invalid argument").
        assert gemini_generation_config("gemini-3.5-flash-lite")["thinkingConfig"] == {
            "thinkingLevel": "minimal"
        }
        assert gemini_generation_config("gemini-3.5-flash")["thinkingConfig"] == {
            "thinkingLevel": "low"
        }

    def test_chat_sends_generation_config(self, client: TestClient, gemini_settings):
        captured = []

        async def mock_stream(payload, model):
            captured.append((payload, model))
            yield "ok"

        with patch("app.routes.ai.ai.stream_gemini", new=mock_stream):
            client.post("/ai/chat", json={**CHAT, "model": "gemini-2.5-flash"})

        payload, model = captured[0]
        assert model == STRONG
        assert payload["generationConfig"] == gemini_generation_config(model)


# ─── Token usage ───────────────────────────────────────────────────────────────


class TestUsage:
    def test_parser_reads_usage_and_skips_thoughts(self):
        parser = GeminiStreamParser()
        chunk = json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "pensando…", "thought": True},
                                {"text": "Respuesta"},
                            ]
                        }
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 120,
                    "candidatesTokenCount": 8,
                    "thoughtsTokenCount": 30,
                },
            }
        )
        assert list(parser.feed("[" + chunk + "]")) == ["Respuesta"]
        assert parser.usage["promptTokenCount"] == 120

    def test_usage_trailer_when_asked(self, client: TestClient, gemini_settings):
        async def mock_stream(payload, model, usage):
            usage.update(provider="gemini", input=100, output=10)
            yield "Hola"

        with patch("app.routes.ai.ai.stream_gemini", new=mock_stream):
            response = client.post("/ai/chat", json={**CHAT, "include_usage": True})

        text, trailer = response.text.split(USAGE_MARKER)
        assert text == "Hola"
        assert json.loads(trailer) == {"provider": "gemini", "input": 100, "output": 10}

    def test_no_trailer_by_default(self, client: TestClient, gemini_settings):
        async def mock_stream(payload, model):
            yield "Hola"

        with patch("app.routes.ai.ai.stream_gemini", new=mock_stream):
            response = client.post("/ai/chat", json=CHAT)

        assert response.text == "Hola"
        assert USAGE_MARKER not in response.text

    @pytest.mark.asyncio
    async def test_claude_usage_from_sse(self, monkeypatch):
        monkeypatch.setattr("app.routes.ai.ai.settings.ANTHROPIC_API_KEY", "key")
        lines = [
            "data: "
            + json.dumps(
                {
                    "type": "message_start",
                    "message": {
                        "usage": {
                            "input_tokens": 50,
                            "cache_read_input_tokens": 900,
                            "cache_creation_input_tokens": 0,
                        }
                    },
                }
            ),
            "data: "
            + json.dumps({"type": "content_block_delta", "delta": {"text": "Hi"}}),
            "data: "
            + json.dumps({"type": "message_delta", "usage": {"output_tokens": 4}}),
        ]
        mock_ctx, _ = make_httpx_mock(200, body_lines=lines)
        usage: dict = {}
        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_ctx):
            chunks = await collect_stream(
                stream_claude({"model": CLAUDE_MODELS["haiku"]}, usage=usage)
            )

        assert chunks == ["Hi"]
        assert usage["input"] == 50
        assert usage["cached"] == 900
        assert usage["output"] == 4
        assert usage["model"] == CLAUDE_MODELS["haiku"]


# ─── Claude ────────────────────────────────────────────────────────────────────


class TestClaude:
    def test_system_prompt_is_cached(self, client: TestClient, anthropic_settings):
        captured = []

        async def mock_stream(payload):
            captured.append(payload)
            yield "ok"

        with patch("app.routes.ai.ai.stream_claude", new=mock_stream):
            client.post("/ai/chat", json=CHAT)

        payload = captured[0]
        assert payload["max_tokens"] == settings.MAX_OUTPUT_TOKENS
        assert payload["system"] == [
            {
                "type": "text",
                "text": "Eres Lumina.",
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def test_only_the_stable_part_is_cached(
        self, client: TestClient, anthropic_settings
    ):
        captured = []

        async def mock_stream(payload):
            captured.append(payload)
            yield "ok"

        with patch("app.routes.ai.ai.stream_claude", new=mock_stream):
            client.post(
                "/ai/chat",
                json={
                    **CHAT,
                    "system_context": "ESTABLE|hora 10:00",
                    "cache_prefix_chars": len("ESTABLE|"),
                },
            )

        assert captured[0]["system"] == [
            {
                "type": "text",
                "text": "ESTABLE|",
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": "hora 10:00"},
        ]

    def test_falls_back_to_gemini_when_claude_fails(
        self, client: TestClient, anthropic_settings, monkeypatch
    ):
        monkeypatch.setattr(
            "app.routes.ai.ai.settings.GOOGLE_GENERATIVE_AI_API_KEY", "gemini-key"
        )

        async def failing_claude(payload):
            yield TOO_MANY_REQUESTS

        async def gemini(payload, model):
            yield "Respuesta de Gemini"

        with (
            patch("app.routes.ai.ai.stream_claude", new=failing_claude),
            patch("app.routes.ai.ai.stream_gemini", new=gemini),
        ):
            response = client.post("/ai/chat", json=CHAT)

        assert response.text == "Respuesta de Gemini"

    def test_shows_claude_error_without_gemini(
        self, client: TestClient, anthropic_settings
    ):
        async def failing_claude(payload):
            yield TOO_MANY_REQUESTS

        with patch("app.routes.ai.ai.stream_claude", new=failing_claude):
            response = client.post("/ai/chat", json=CHAT)

        assert response.text == TOO_MANY_REQUESTS


# ─── Internal token ────────────────────────────────────────────────────────────


class TestInternalToken:
    @pytest.fixture
    def token(self, monkeypatch):
        monkeypatch.setattr("app.security.settings.INTERNAL_TOKEN", "s3cret")
        return "s3cret"

    def test_rejects_missing_token(self, client: TestClient, token, gemini_settings):
        assert client.post("/ai/chat", json=CHAT).status_code == 401

    def test_rejects_wrong_token(self, client: TestClient, token, gemini_settings):
        response = client.post(
            "/ai/chat", json=CHAT, headers={"X-Internal-Token": "nope"}
        )
        assert response.status_code == 401

    def test_planner_is_protected_too(self, client: TestClient, token):
        assert client.post("/ai/planner/organize", json={}).status_code == 401

    def test_accepts_the_token(self, client: TestClient, token, gemini_settings):
        async def mock_stream(payload, model):
            yield "ok"

        with patch("app.routes.ai.ai.stream_gemini", new=mock_stream):
            response = client.post(
                "/ai/chat", json=CHAT, headers={"X-Internal-Token": token}
            )
        assert response.status_code == 200

    def test_health_stays_public(self, client: TestClient, token):
        assert client.get("/health").status_code == 200
        assert client.get("/").status_code == 200
