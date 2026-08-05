"""
tests/test_routes_ai.py

Tests funcionales del router /ai (chat + analyze-patterns).
Estrategia: mock de httpx y settings para aislar la lógica
de routing/serialización de las llamadas reales a la API.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


# ═══════════════════════════════════════════════════════════════════════════════
# 1. POST /ai/chat — Validación de request
# ═══════════════════════════════════════════════════════════════════════════════


class TestChatRequestValidation:
    """Valida que el endpoint rechace payloads malformados antes de llamar a la API."""

    def test_empty_messages_returns_400(self, client: TestClient):
        response = client.post(
            "/ai/chat",
            json={"messages": [], "system_context": "Eres un asistente."},
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_missing_messages_field_returns_422(self, client: TestClient):
        response = client.post(
            "/ai/chat",
            json={"system_context": "Eres un asistente."},
        )
        assert response.status_code == 422

    def test_missing_system_context_returns_422(self, client: TestClient):
        response = client.post(
            "/ai/chat",
            json={"messages": [{"role": "user", "content": "Hola"}]},
        )
        assert response.status_code == 422

    def test_invalid_message_role_is_accepted(
        self, client: TestClient, gemini_settings
    ):
        """El backend no valida roles en la capa Pydantic — pasan como string."""

        async def mock_stream(payload, model):
            yield "OK"

        with patch("app.routes.ai.ai.stream_gemini", new=mock_stream):
            response = client.post(
                "/ai/chat",
                json={
                    "messages": [{"role": "invalid_role", "content": "test"}],
                    "system_context": "ctx",
                },
            )
            # Debe llegar hasta el streamer, no rechazarse en validación
            assert response.status_code == 200

    def test_message_with_empty_content_is_accepted(
        self, client: TestClient, gemini_settings
    ):
        """Contenido vacío es válido en Pydantic pero el modelo lo manejará."""

        async def mock_stream(payload, model):
            yield ""

        with patch("app.routes.ai.ai.stream_gemini", new=mock_stream):
            response = client.post(
                "/ai/chat",
                json={
                    "messages": [{"role": "user", "content": ""}],
                    "system_context": "ctx",
                },
            )
            assert response.status_code == 200

    def test_model_field_defaults_to_gemini(self, client: TestClient, gemini_settings):
        """Sin campo model, debe usar gemini-2.5-flash-lite."""
        captured_model = []

        async def mock_stream(payload, model):
            captured_model.append(model)
            yield "ok"

        with patch("app.routes.ai.ai.stream_gemini", new=mock_stream):
            client.post(
                "/ai/chat",
                json={
                    "messages": [{"role": "user", "content": "Hola"}],
                    "system_context": "ctx",
                },
            )
        assert captured_model[0] == "gemini-2.5-flash-lite"

    def test_model_none_uses_default(self, client: TestClient, gemini_settings):
        """model: null debe caer en el fallback gemini-2.5-flash-lite."""
        captured_model = []

        async def mock_stream(payload, model):
            captured_model.append(model)
            yield "ok"

        with patch("app.routes.ai.ai.stream_gemini", new=mock_stream):
            client.post(
                "/ai/chat",
                json={
                    "messages": [{"role": "user", "content": "Hola"}],
                    "system_context": "ctx",
                    "model": None,
                },
            )
        assert captured_model[0] == "gemini-2.5-flash-lite"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. POST /ai/chat — Comportamiento con Gemini (sin ANTHROPIC_API_KEY)
# ═══════════════════════════════════════════════════════════════════════════════


class TestChatGeminiBehavior:
    """Valida la lógica de routing hacia Gemini cuando no hay clave Anthropic."""

    def test_single_message_uses_gemini_stream(
        self, client: TestClient, gemini_settings
    ):
        captured_payload = []

        async def mock_stream(payload, model):
            captured_payload.append(payload)
            yield "Hola, soy Lumina"

        with patch("app.routes.ai.ai.stream_gemini", new=mock_stream):
            response = client.post(
                "/ai/chat",
                json={
                    "messages": [{"role": "user", "content": "Ayúdame a planificar"}],
                    "system_context": "Eres un asistente productivo.",
                },
            )
        assert response.status_code == 200
        assert "Hola, soy Lumina" in response.text

    def test_chat_history_is_sent_in_gemini_contents(
        self, client: TestClient, gemini_settings
    ):
        """Mensajes de historial deben incluirse en contents antes del último."""
        captured_payload = []

        async def mock_stream(payload, model):
            captured_payload.append(payload)
            yield "ok"

        with patch("app.routes.ai.ai.stream_gemini", new=mock_stream):
            client.post(
                "/ai/chat",
                json={
                    "messages": [
                        {"role": "user", "content": "msg1"},
                        {"role": "assistant", "content": "resp1"},
                        {"role": "user", "content": "msg2"},
                    ],
                    "system_context": "ctx",
                },
            )
        payload = captured_payload[0]
        # 2 mensajes en historial (todos menos el último) + el último
        assert len(payload["contents"]) == 3

    def test_gemini_role_mapping_model_not_user(
        self, client: TestClient, gemini_settings
    ):
        """Rol 'assistant' debe mapearse a 'model' en el payload de Gemini."""
        captured_payload = []

        async def mock_stream(payload, model):
            captured_payload.append(payload)
            yield "ok"

        with patch("app.routes.ai.ai.stream_gemini", new=mock_stream):
            client.post(
                "/ai/chat",
                json={
                    "messages": [
                        {"role": "assistant", "content": "Soy un asistente"},
                        {"role": "user", "content": "Gracias"},
                    ],
                    "system_context": "ctx",
                },
            )
        contents = captured_payload[0]["contents"]
        assert contents[0]["role"] == "model"

    def test_system_instruction_included_in_payload(
        self, client: TestClient, gemini_settings
    ):
        captured_payload = []

        async def mock_stream(payload, model):
            captured_payload.append(payload)
            yield "ok"

        with patch("app.routes.ai.ai.stream_gemini", new=mock_stream):
            client.post(
                "/ai/chat",
                json={
                    "messages": [{"role": "user", "content": "Hola"}],
                    "system_context": "Eres Lumina, asistente productivo.",
                },
            )
        payload = captured_payload[0]
        system_text = payload["systemInstruction"]["parts"][0]["text"]
        assert "Lumina" in system_text

    def test_claude_model_in_name_redirected_to_gemini(
        self, client: TestClient, gemini_settings
    ):
        """Si model contiene 'claude' pero no hay ANTHROPIC_API_KEY, usa Gemini."""
        captured_model = []

        async def mock_stream(payload, model):
            captured_model.append(model)
            yield "ok"

        with patch("app.routes.ai.ai.stream_gemini", new=mock_stream):
            client.post(
                "/ai/chat",
                json={
                    "messages": [{"role": "user", "content": "Hola"}],
                    "system_context": "ctx",
                    "model": "claude-sonnet",
                },
            )
        # Gemini se llama con "claude-sonnet" — la redirección ocurre dentro de stream_gemini
        assert len(captured_model) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 3. POST /ai/chat — Comportamiento con Claude (con ANTHROPIC_API_KEY)
# ═══════════════════════════════════════════════════════════════════════════════


class TestChatClaudeBehavior:
    """Valida que la rama Claude se active cuando ANTHROPIC_API_KEY está presente."""

    def test_anthropic_key_activates_claude_stream(
        self, client: TestClient, anthropic_settings
    ):
        captured = []

        async def mock_stream(payload):
            captured.append(payload)
            yield "Claude response"

        with patch("app.routes.ai.ai.stream_claude", new=mock_stream):
            response = client.post(
                "/ai/chat",
                json={
                    "messages": [{"role": "user", "content": "Hola"}],
                    "system_context": "ctx",
                },
            )
        assert response.status_code == 200
        assert "Claude response" in response.text

    def test_claude_model_sonnet_mapping(self, client: TestClient, anthropic_settings):
        captured = []

        async def mock_stream(payload):
            captured.append(payload)
            yield "ok"

        with patch("app.routes.ai.ai.stream_claude", new=mock_stream):
            client.post(
                "/ai/chat",
                json={
                    "messages": [{"role": "user", "content": "Hola"}],
                    "system_context": "ctx",
                    "model": "sonnet",
                },
            )
        assert captured[0]["model"] == "claude-sonnet-5"

    def test_claude_model_haiku_mapping(self, client: TestClient, anthropic_settings):
        captured = []

        async def mock_stream(payload):
            captured.append(payload)
            yield "ok"

        with patch("app.routes.ai.ai.stream_claude", new=mock_stream):
            client.post(
                "/ai/chat",
                json={
                    "messages": [{"role": "user", "content": "Hola"}],
                    "system_context": "ctx",
                    "model": "haiku",
                },
            )
        assert captured[0]["model"] == "claude-haiku-4-5-20251001"

    def test_claude_model_opus_mapping(self, client: TestClient, anthropic_settings):
        captured = []

        async def mock_stream(payload):
            captured.append(payload)
            yield "ok"

        with patch("app.routes.ai.ai.stream_claude", new=mock_stream):
            client.post(
                "/ai/chat",
                json={
                    "messages": [{"role": "user", "content": "Hola"}],
                    "system_context": "ctx",
                    "model": "opus",
                },
            )
        assert captured[0]["model"] == "claude-opus-4-8"

    def test_claude_unknown_model_falls_back_to_sonnet(
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
                    "messages": [{"role": "user", "content": "Hola"}],
                    "system_context": "ctx",
                    "model": "some-unknown-model",
                },
            )
        assert captured[0]["model"] == "claude-sonnet-5"

    def test_claude_payload_includes_stream_true(
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
                    "messages": [{"role": "user", "content": "Hola"}],
                    "system_context": "ctx",
                },
            )
        assert captured[0]["stream"] is True

    def test_claude_payload_max_tokens_is_set(
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
                    "messages": [{"role": "user", "content": "Hola"}],
                    "system_context": "ctx",
                },
            )
        assert "max_tokens" in captured[0]
        assert captured[0]["max_tokens"] > 0

    def test_claude_user_message_role_mapping(
        self, client: TestClient, anthropic_settings
    ):
        """Rol 'user' debe quedar como 'user'; 'assistant' también debe ser 'assistant'."""
        captured = []

        async def mock_stream(payload):
            captured.append(payload)
            yield "ok"

        with patch("app.routes.ai.ai.stream_claude", new=mock_stream):
            client.post(
                "/ai/chat",
                json={
                    "messages": [
                        {"role": "user", "content": "Hola"},
                        {"role": "assistant", "content": "Hola de vuelta"},
                        {"role": "user", "content": "¿Cómo estás?"},
                    ],
                    "system_context": "ctx",
                },
            )
        messages = captured[0]["messages"]
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. POST /ai/analyze-patterns — Validación de request
# ═══════════════════════════════════════════════════════════════════════════════


class TestAnalyzePatternsValidation:
    """Valida que el endpoint de análisis de patrones rechace inputs inválidos."""

    def test_missing_user_name_returns_422(self, client: TestClient):
        response = client.post(
            "/ai/analyze-patterns",
            json={
                "hour_buckets": {},
                "task_stats": {},
                "session_stats": {},
                "top_productive_hours": [],
                "work_style_hint": "morning",
            },
        )
        assert response.status_code == 422

    def test_missing_hour_buckets_returns_422(
        self, client: TestClient, sample_patterns_payload
    ):
        payload = {
            k: v for k, v in sample_patterns_payload.items() if k != "hour_buckets"
        }
        response = client.post("/ai/analyze-patterns", json=payload)
        assert response.status_code == 422

    def test_top_productive_hours_must_be_list(
        self, client: TestClient, sample_patterns_payload
    ):
        payload = {**sample_patterns_payload, "top_productive_hours": "morning"}
        response = client.post("/ai/analyze-patterns", json=payload)
        assert response.status_code == 422

    def test_no_gemini_key_returns_500(
        self, client: TestClient, no_api_keys, sample_patterns_payload
    ):
        response = client.post("/ai/analyze-patterns", json=sample_patterns_payload)
        assert response.status_code == 500
        assert "Gemini API key" in response.json()["detail"]

    def test_empty_hour_buckets_is_valid(
        self, client: TestClient, gemini_settings, sample_patterns_payload
    ):
        """Buckets vacíos son un caso real de usuario nuevo sin historial."""
        payload = {**sample_patterns_payload, "hour_buckets": {}}
        gemini_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "goldenHours": "09:00 - 11:00",
                                        "goldenHoursConfidence": 0.5,
                                        "behaviorSummary": "Aún no hay suficientes datos.",
                                        "patterns": [],
                                        "workStyle": "En construcción",
                                    }
                                )
                            }
                        ]
                    }
                }
            ]
        }

        async def mock_post(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = gemini_response
            return mock_resp

        with patch("httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            instance.post = mock_post
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            response = client.post("/ai/analyze-patterns", json=payload)
        assert response.status_code == 200


class TestAnalyzePatternsSuccess:
    """Valida el flujo exitoso del endpoint de análisis de patrones."""

    def _mock_gemini_response(self, data: dict) -> dict:
        return {"candidates": [{"content": {"parts": [{"text": json.dumps(data)}]}}]}

    def test_successful_response_structure(
        self, client: TestClient, gemini_settings, sample_patterns_payload
    ):
        expected_data = {
            "goldenHours": "22:00 - 00:00",
            "goldenHoursConfidence": 0.92,
            "behaviorSummary": "¡Alexis, eres un experto nocturno!",
            "patterns": [{"label": "Madrugador Estrella 🌟", "icon": "🌟"}],
            "workStyle": "Desarrollador de Élite 💻",
        }
        gemini_response = self._mock_gemini_response(expected_data)

        async def mock_post(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = gemini_response
            return mock_resp

        with patch("httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            instance.post = mock_post
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            response = client.post("/ai/analyze-patterns", json=sample_patterns_payload)

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["data"]["goldenHours"] == "22:00 - 00:00"

    def test_gemini_502_returns_502_to_client(
        self, client: TestClient, gemini_settings, sample_patterns_payload
    ):
        async def mock_post(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 503
            mock_resp.json.return_value = {}
            return mock_resp

        with patch("httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            instance.post = mock_post
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            response = client.post("/ai/analyze-patterns", json=sample_patterns_payload)

        assert response.status_code in (502, 500)

    def test_user_name_injected_into_gemini_payload(
        self, client: TestClient, gemini_settings, sample_patterns_payload
    ):
        """El nombre de usuario debe aparecer en el prompt enviado a Gemini."""
        captured_payload = []

        expected_data = {
            "goldenHours": "09:00 - 11:00",
            "goldenHoursConfidence": 0.8,
            "behaviorSummary": "¡Alexis!",
            "patterns": [],
            "workStyle": "Estratega",
        }

        async def mock_post(url, json=None, timeout=None):
            if json:
                captured_payload.append(json)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "candidates": [
                    {"content": {"parts": [{"text": json_dumps(expected_data)}]}}
                ]
            }
            return mock_resp

        def json_dumps(d):
            return __import__("json").dumps(d)

        with patch("httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            instance.post = mock_post
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            client.post("/ai/analyze-patterns", json=sample_patterns_payload)

        # El nombre del usuario debe estar en el contenido del prompt
        assert len(captured_payload) > 0
        content_text = captured_payload[0]["contents"][0]["parts"][0]["text"]
        assert "Alexis" in content_text


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GeminiStreamParser — Tests unitarios del parser de streaming
# ═══════════════════════════════════════════════════════════════════════════════


class TestGeminiStreamParser:
    """Tests de la lógica del parser de chunks JSON de Gemini."""

    def _get_parser(self):
        from app.routes.ai.ai import GeminiStreamParser

        return GeminiStreamParser()

    def _build_chunk(self, text: str) -> str:
        return json.dumps({"candidates": [{"content": {"parts": [{"text": text}]}}]})

    def test_parser_extracts_text_from_single_chunk(self):
        parser = self._get_parser()
        chunk = self._build_chunk("Hola, soy Lumina")
        results = list(parser.feed(chunk))
        assert results == ["Hola, soy Lumina"]

    def test_parser_handles_empty_chunk(self):
        parser = self._get_parser()
        results = list(parser.feed(""))
        assert results == []

    def test_parser_handles_json_array_wrapper(self):
        """La API de Gemini envuelve los chunks en un array JSON."""
        parser = self._get_parser()
        chunk = "[" + self._build_chunk("texto") + "]"
        results = list(parser.feed(chunk))
        assert "texto" in results

    def test_parser_handles_malformed_json_gracefully(self):
        parser = self._get_parser()
        results = list(parser.feed("{malformed json}"))
        assert results == []

    def test_parser_handles_chunk_without_candidates(self):
        parser = self._get_parser()
        chunk = json.dumps({"other": "data"})
        results = list(parser.feed(chunk))
        assert results == []

    def test_parser_handles_empty_candidates_list(self):
        parser = self._get_parser()
        chunk = json.dumps({"candidates": []})
        results = list(parser.feed(chunk))
        assert results == []

    def test_parser_handles_empty_parts(self):
        parser = self._get_parser()
        chunk = json.dumps({"candidates": [{"content": {"parts": []}}]})
        results = list(parser.feed(chunk))
        assert results == []

    def test_parser_handles_empty_text_field(self):
        parser = self._get_parser()
        chunk = json.dumps({"candidates": [{"content": {"parts": [{"text": ""}]}}]})
        results = list(parser.feed(chunk))
        assert results == []

    def test_parser_handles_multiple_objects_in_one_feed(self):
        """Gemini puede enviar múltiples objetos en un solo chunk de streaming."""
        parser = self._get_parser()
        chunk1 = self._build_chunk("Primera ")
        chunk2 = self._build_chunk("Segunda")
        combined = chunk1 + "\n" + chunk2
        results = list(parser.feed(combined))
        assert len(results) == 2
        assert "Primera " in results
        assert "Segunda" in results

    def test_parser_state_persists_between_feeds(self):
        """El buffer debe acumular chunks parciales."""
        parser = self._get_parser()
        full_chunk = self._build_chunk("completo")
        # Dividir el chunk en dos partes
        half = len(full_chunk) // 2
        list(parser.feed(full_chunk[:half]))
        results = list(parser.feed(full_chunk[half:]))
        assert "completo" in results

    def test_parser_handles_unicode_text(self):
        parser = self._get_parser()
        chunk = self._build_chunk("¡Hola! 🌟 Productividad élite")
        results = list(parser.feed(chunk))
        assert "🌟" in results[0]

    def test_parser_handles_escaped_quotes_in_text(self):
        parser = self._get_parser()
        text = 'Usa el formato "ISO 8601" para fechas'
        chunk = self._build_chunk(text)
        results = list(parser.feed(chunk))
        assert text in results


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


async def async_gen(items: list):
    """Generador asíncrono para simular streamers."""
    for item in items:
        yield item
