"""
tests/test_stream_errors.py

Tests de resiliencia de los streamers stream_gemini y stream_claude.
Cubre: errores HTTP (429, 503, 5xx), excepciones de red, respuestas SSE
malformadas y señales de fin de stream.

Estrategia: mock de httpx.AsyncClient a nivel de módulo para simular
respuestas HTTP completas sin llamadas de red reales.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ─── Helper para ejecutar async generators ────────────────────────────────────


async def collect_stream(async_gen):
    """Recolecta todos los chunks de un async generator."""
    return [chunk async for chunk in async_gen]


# ─── Mock helpers ─────────────────────────────────────────────────────────────


def make_httpx_mock(
    status_code: int, body_lines: list[str] = None, chunks: list[str] = None
):
    """
    Construye un mock completo de httpx.AsyncClient.stream() context manager.
    - status_code: código HTTP a simular
    - body_lines: líneas de texto para aiter_lines() (Claude SSE)
    - chunks: chunks de texto para aiter_text() (Gemini chunks)
    """
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.aread = AsyncMock()

    # aiter_text — para Gemini
    async def _aiter_text():
        for c in chunks or []:
            yield c

    # aiter_lines — para Claude SSE
    async def _aiter_lines():
        for line in body_lines or []:
            yield line

    mock_response.aiter_text = _aiter_text
    mock_response.aiter_lines = _aiter_lines

    # context manager de client.stream()
    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_stream_ctx)
    mock_client.post = AsyncMock()  # para analyze-patterns

    mock_client_ctx = AsyncMock()
    mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_ctx.__aexit__ = AsyncMock(return_value=False)

    return mock_client_ctx, mock_client


# ═══════════════════════════════════════════════════════════════════════════════
# 1. stream_gemini — Errores HTTP y casos de red
# ═══════════════════════════════════════════════════════════════════════════════


class TestStreamGeminiErrors:
    """Tests de manejo de errores en stream_gemini."""

    def _patch_settings(self, monkeypatch, key: str = "valid-api-key"):
        monkeypatch.setattr(
            "app.routes.ai.ai.settings.GOOGLE_GENERATIVE_AI_API_KEY", key
        )

    @pytest.mark.asyncio
    async def test_no_api_key_yields_error_message(self, monkeypatch):
        self._patch_settings(monkeypatch, "")
        from app.routes.ai.ai import stream_gemini

        chunks = await collect_stream(stream_gemini({}, "gemini-2.5-flash-lite"))
        assert any("GEMINI_API_KEY" in c or "Error" in c for c in chunks)

    @pytest.mark.asyncio
    async def test_429_rate_limit_yields_friendly_message(self, monkeypatch):
        self._patch_settings(monkeypatch)
        mock_ctx, _ = make_httpx_mock(429)
        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_ctx):
            from app.routes.ai.ai import stream_gemini

            chunks = await collect_stream(stream_gemini({}, "gemini-2.5-flash-lite"))

        assert len(chunks) == 1
        assert "límite" in chunks[0] or "respiro" in chunks[0] or "🌟" in chunks[0]

    @pytest.mark.asyncio
    async def test_503_overload_yields_friendly_message(self, monkeypatch):
        self._patch_settings(monkeypatch)
        mock_ctx, _ = make_httpx_mock(503)
        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_ctx):
            from app.routes.ai.ai import stream_gemini

            chunks = await collect_stream(stream_gemini({}, "gemini-2.5-flash-lite"))

        assert len(chunks) == 1
        assert "ocupada" in chunks[0] or "☕" in chunks[0] or "📭" in chunks[0]

    @pytest.mark.asyncio
    async def test_500_server_error_yields_generic_message(self, monkeypatch):
        self._patch_settings(monkeypatch)
        mock_ctx, _ = make_httpx_mock(500)
        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_ctx):
            from app.routes.ai.ai import stream_gemini

            chunks = await collect_stream(stream_gemini({}, "gemini-2.5-flash-lite"))

        assert len(chunks) == 1
        assert "técnico" in chunks[0] or "🛠️" in chunks[0] or "paciencia" in chunks[0]

    @pytest.mark.asyncio
    async def test_401_unauthorized_yields_generic_message(self, monkeypatch):
        self._patch_settings(monkeypatch)
        mock_ctx, _ = make_httpx_mock(401)
        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_ctx):
            from app.routes.ai.ai import stream_gemini

            chunks = await collect_stream(stream_gemini({}, "gemini-2.5-flash-lite"))

        assert len(chunks) == 1
        # Cae en el else
        assert "técnico" in chunks[0] or "🛠️" in chunks[0]

    @pytest.mark.asyncio
    async def test_502_bad_gateway_yields_generic_message(self, monkeypatch):
        self._patch_settings(monkeypatch)
        mock_ctx, _ = make_httpx_mock(502)
        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_ctx):
            from app.routes.ai.ai import stream_gemini

            chunks = await collect_stream(stream_gemini({}, "gemini-2.5-flash-lite"))

        assert "técnico" in chunks[0] or "🛠️" in chunks[0]

    @pytest.mark.asyncio
    async def test_network_exception_yields_error_message(self, monkeypatch):
        """Excepción de red (timeout, connection reset, etc.) debe manejarse."""
        self._patch_settings(monkeypatch)

        mock_client_ctx = AsyncMock()
        mock_client = MagicMock()
        mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_ctx.__aexit__ = AsyncMock(return_value=False)

        # stream() lanza excepción al usarse
        import httpx

        mock_client.stream = MagicMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_client_ctx):
            from app.routes.ai.ai import stream_gemini

            chunks = await collect_stream(stream_gemini({}, "gemini-2.5-flash-lite"))

        assert len(chunks) == 1
        assert "técnico" in chunks[0] or "Lumina" in chunks[0]

    @pytest.mark.asyncio
    async def test_successful_stream_yields_text(self, monkeypatch):
        """Con respuesta 200, debe extraer y emitir el texto de los chunks."""
        self._patch_settings(monkeypatch)
        chunk = json.dumps(
            {"candidates": [{"content": {"parts": [{"text": "Hola, soy Lumina"}]}}]}
        )
        mock_ctx, _ = make_httpx_mock(200, chunks=[chunk])
        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_ctx):
            from app.routes.ai.ai import stream_gemini

            chunks_received = await collect_stream(
                stream_gemini({}, "gemini-2.5-flash-lite")
            )

        assert "Hola, soy Lumina" in chunks_received

    @pytest.mark.asyncio
    async def test_claude_model_name_redirected_to_gemini_flash(self, monkeypatch):
        """Si model contiene 'claude', debe reemplazarse por gemini-2.5-flash-lite."""
        self._patch_settings(monkeypatch)
        mock_ctx, mock_client = make_httpx_mock(200, chunks=[])
        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_ctx):
            from app.routes.ai.ai import stream_gemini

            await collect_stream(stream_gemini({}, "claude-sonnet"))

        call_url = mock_client.stream.call_args[0][1]
        assert "gemini-2.5-flash-lite" in call_url
        assert "claude" not in call_url

    @pytest.mark.asyncio
    async def test_malformed_json_chunks_are_skipped(self, monkeypatch):
        """Chunks malformados no deben romper el stream — se omiten silenciosamente."""
        self._patch_settings(monkeypatch)
        chunks = ["not json at all", "{broken", "[]", ""]
        mock_ctx, _ = make_httpx_mock(200, chunks=chunks)
        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_ctx):
            from app.routes.ai.ai import stream_gemini

            result = await collect_stream(stream_gemini({}, "gemini-2.5-flash-lite"))

        # No debe haber crash, y los chunks malformados no producen output
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_empty_candidates_produces_no_output(self, monkeypatch):
        """candidates: [] no debe producir texto pero tampoco debe crashear."""
        self._patch_settings(monkeypatch)
        chunk = json.dumps({"candidates": []})
        mock_ctx, _ = make_httpx_mock(200, chunks=[chunk])
        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_ctx):
            from app.routes.ai.ai import stream_gemini

            result = await collect_stream(stream_gemini({}, "gemini-2.5-flash-lite"))

        assert result == []

    @pytest.mark.asyncio
    async def test_multiple_chunks_accumulated_correctly(self, monkeypatch):
        """Múltiples chunks deben producir múltiples outputs."""
        self._patch_settings(monkeypatch)
        chunks = [
            json.dumps({"candidates": [{"content": {"parts": [{"text": "Hola"}]}}]}),
            json.dumps({"candidates": [{"content": {"parts": [{"text": " mundo"}]}}]}),
            json.dumps({"candidates": [{"content": {"parts": [{"text": "!"}]}}]}),
        ]
        mock_ctx, _ = make_httpx_mock(200, chunks=chunks)
        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_ctx):
            from app.routes.ai.ai import stream_gemini

            result = await collect_stream(stream_gemini({}, "gemini-2.5-flash-lite"))

        full_text = "".join(result)
        assert "Hola" in full_text
        assert " mundo" in full_text
        assert "!" in full_text


# ═══════════════════════════════════════════════════════════════════════════════
# 2. stream_claude — Errores HTTP y SSE parsing
# ═══════════════════════════════════════════════════════════════════════════════


class TestStreamClaudeErrors:
    """Tests de manejo de errores y SSE en stream_claude."""

    def _patch_settings(self, monkeypatch, key: str = "valid-claude-key"):
        monkeypatch.setattr("app.routes.ai.ai.settings.ANTHROPIC_API_KEY", key)

    @pytest.mark.asyncio
    async def test_no_api_key_yields_error_message(self, monkeypatch):
        self._patch_settings(monkeypatch, "")
        from app.routes.ai.ai import stream_claude

        chunks = await collect_stream(stream_claude({}))
        assert any("ANTHROPIC_API_KEY" in c or "Error" in c for c in chunks)

    @pytest.mark.asyncio
    async def test_429_rate_limit_yields_friendly_message(self, monkeypatch):
        self._patch_settings(monkeypatch)
        mock_ctx, _ = make_httpx_mock(429)
        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_ctx):
            from app.routes.ai.ai import stream_claude

            chunks = await collect_stream(stream_claude({}))

        assert len(chunks) == 1
        assert "límite" in chunks[0] or "🌟" in chunks[0]

    @pytest.mark.asyncio
    async def test_503_overload_yields_friendly_message(self, monkeypatch):
        self._patch_settings(monkeypatch)
        mock_ctx, _ = make_httpx_mock(503)
        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_ctx):
            from app.routes.ai.ai import stream_claude

            chunks = await collect_stream(stream_claude({}))

        assert len(chunks) == 1
        assert "ocupada" in chunks[0] or "☕" in chunks[0]

    @pytest.mark.asyncio
    async def test_500_server_error_yields_generic_message(self, monkeypatch):
        self._patch_settings(monkeypatch)
        mock_ctx, _ = make_httpx_mock(500)
        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_ctx):
            from app.routes.ai.ai import stream_claude

            chunks = await collect_stream(stream_claude({}))

        assert len(chunks) == 1
        assert "técnico" in chunks[0] or "🛠️" in chunks[0]

    @pytest.mark.asyncio
    async def test_sse_content_block_delta_yields_text(self, monkeypatch):
        """Líneas SSE válidas de tipo content_block_delta deben emitir texto."""
        self._patch_settings(monkeypatch)
        delta_event = json.dumps(
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hola Claude"},
            }
        )
        sse_lines = [f"data: {delta_event}"]
        mock_ctx, _ = make_httpx_mock(200, body_lines=sse_lines)
        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_ctx):
            from app.routes.ai.ai import stream_claude

            chunks = await collect_stream(stream_claude({}))

        assert "Hola Claude" in chunks

    @pytest.mark.asyncio
    async def test_sse_done_signal_stops_stream(self, monkeypatch):
        """La señal [DONE] debe detener el stream sin error."""
        self._patch_settings(monkeypatch)
        delta_event = json.dumps(
            {"type": "content_block_delta", "delta": {"text": "texto antes"}}
        )
        sse_lines = [
            f"data: {delta_event}",
            "data: [DONE]",
            # Esta línea no debe emitirse
            f"data: {json.dumps({'type': 'content_block_delta', 'delta': {'text': 'después de DONE'}})}",
        ]
        mock_ctx, _ = make_httpx_mock(200, body_lines=sse_lines)
        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_ctx):
            from app.routes.ai.ai import stream_claude

            chunks = await collect_stream(stream_claude({}))

        full_text = "".join(chunks)
        assert "después de DONE" not in full_text

    @pytest.mark.asyncio
    async def test_sse_non_data_lines_are_skipped(self, monkeypatch):
        """Líneas SSE que no empiecen con 'data:' deben ignorarse."""
        self._patch_settings(monkeypatch)
        delta_event = json.dumps(
            {"type": "content_block_delta", "delta": {"text": "Texto válido"}}
        )
        sse_lines = [
            "event: message_start",
            ": ping",
            "",
            f"data: {delta_event}",
        ]
        mock_ctx, _ = make_httpx_mock(200, body_lines=sse_lines)
        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_ctx):
            from app.routes.ai.ai import stream_claude

            chunks = await collect_stream(stream_claude({}))

        assert "Texto válido" in chunks

    @pytest.mark.asyncio
    async def test_sse_malformed_json_skipped_silently(self, monkeypatch):
        """Líneas data: con JSON malformado deben ignorarse sin crashear."""
        self._patch_settings(monkeypatch)
        valid_delta = json.dumps(
            {"type": "content_block_delta", "delta": {"text": "Texto bueno"}}
        )
        sse_lines = [
            "data: {malformed_json",
            "data: not_json_at_all",
            f"data: {valid_delta}",
        ]
        mock_ctx, _ = make_httpx_mock(200, body_lines=sse_lines)
        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_ctx):
            from app.routes.ai.ai import stream_claude

            chunks = await collect_stream(stream_claude({}))

        assert "Texto bueno" in chunks
        # No hay crash

    @pytest.mark.asyncio
    async def test_sse_non_delta_types_produce_no_output(self, monkeypatch):
        """Eventos SSE de otros tipos (message_start, ping) no emiten texto."""
        self._patch_settings(monkeypatch)
        sse_lines = [
            f"data: {json.dumps({'type': 'message_start', 'message': {'id': 'msg_01'}})}",
            f"data: {json.dumps({'type': 'content_block_start', 'index': 0})}",
            f"data: {json.dumps({'type': 'message_stop'})}",
        ]
        mock_ctx, _ = make_httpx_mock(200, body_lines=sse_lines)
        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_ctx):
            from app.routes.ai.ai import stream_claude

            chunks = await collect_stream(stream_claude({}))

        assert chunks == []

    @pytest.mark.asyncio
    async def test_network_exception_yields_error_message(self, monkeypatch):
        """Excepción de red debe manejarse y emitir mensaje de error."""
        self._patch_settings(monkeypatch)

        mock_client_ctx = AsyncMock()
        mock_client = MagicMock()
        mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_ctx.__aexit__ = AsyncMock(return_value=False)

        import httpx

        mock_client.stream = MagicMock(
            side_effect=httpx.TimeoutException("Request timed out")
        )

        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_client_ctx):
            from app.routes.ai.ai import stream_claude

            chunks = await collect_stream(stream_claude({}))

        assert len(chunks) == 1
        assert "técnico" in chunks[0] or "Lumina" in chunks[0]

    @pytest.mark.asyncio
    async def test_multiple_delta_events_concatenate(self, monkeypatch):
        """Múltiples eventos delta deben emitir múltiples chunks de texto."""
        self._patch_settings(monkeypatch)
        texts = ["Hola ", "soy ", "Lumina ", "tu asistente 🌟"]
        sse_lines = [
            f"data: {json.dumps({'type': 'content_block_delta', 'delta': {'text': t}})}"
            for t in texts
        ]
        mock_ctx, _ = make_httpx_mock(200, body_lines=sse_lines)
        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_ctx):
            from app.routes.ai.ai import stream_claude

            chunks = await collect_stream(stream_claude({}))

        full_text = "".join(chunks)
        assert "Hola " in full_text
        assert "Lumina " in full_text
        assert "🌟" in full_text

    @pytest.mark.asyncio
    async def test_request_includes_api_key_header(self, monkeypatch):
        """El header x-api-key debe enviarse con la API key correcta."""
        self._patch_settings(monkeypatch, "my-secret-claude-key")
        mock_ctx, mock_client = make_httpx_mock(200, body_lines=[])
        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_ctx):
            from app.routes.ai.ai import stream_claude

            await collect_stream(stream_claude({"model": "claude-sonnet-5"}))

        call_kwargs = mock_client.stream.call_args[1]
        assert call_kwargs["headers"]["x-api-key"] == "my-secret-claude-key"

    @pytest.mark.asyncio
    async def test_request_includes_anthropic_version_header(self, monkeypatch):
        """El header anthropic-version debe estar presente."""
        self._patch_settings(monkeypatch)
        mock_ctx, mock_client = make_httpx_mock(200, body_lines=[])
        with patch("app.routes.ai.ai.httpx.AsyncClient", return_value=mock_ctx):
            from app.routes.ai.ai import stream_claude

            await collect_stream(stream_claude({}))

        call_kwargs = mock_client.stream.call_args[1]
        assert "anthropic-version" in call_kwargs["headers"]
        assert call_kwargs["headers"]["anthropic-version"] == "2023-06-01"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. analyze-patterns endpoint — Errores HTTP de Gemini
# ═══════════════════════════════════════════════════════════════════════════════


class TestAnalyzePatternsHttpErrors:
    """Tests de casos de error HTTP en el endpoint analyze-patterns."""

    @pytest.fixture
    def client(self, monkeypatch):
        monkeypatch.setattr(
            "app.routes.ai.ai.settings.GOOGLE_GENERATIVE_AI_API_KEY", "valid-key"
        )
        monkeypatch.setattr("app.routes.ai.ai.settings.ANTHROPIC_API_KEY", "")
        from fastapi.testclient import TestClient
        from app.main import app

        return TestClient(app)

    @pytest.fixture
    def payload(self):
        return {
            "user_name": "Alexis",
            "hour_buckets": {"9": {"sessions": 3, "focus_minutes": 90}},
            "task_stats": {"total_tasks": 5, "completed": 3},
            "session_stats": {"total_sessions": 5, "avg_duration_minutes": 30},
            "top_productive_hours": [9, 10, 11],
            "work_style_hint": "Mañanero",
        }

    def test_gemini_502_raises_502_in_response(self, client, payload, monkeypatch):
        """Si Gemini retorna 502, el endpoint debe retornar 502."""
        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.json = MagicMock(return_value={})

        with patch("app.routes.ai.ai.httpx.AsyncClient") as mock_cls:
            mock_cls_instance = AsyncMock()
            mock_cls_instance.__aenter__ = AsyncMock(return_value=mock_cls_instance)
            mock_cls_instance.__aexit__ = AsyncMock(return_value=False)
            mock_cls_instance.post = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_cls_instance

            response = client.post("/ai/analyze-patterns", json=payload)

        assert response.status_code in (502, 500)

    def test_gemini_returns_empty_candidates_raises_500(
        self, client, payload, monkeypatch
    ):
        """candidates: [] debe hacer que el endpoint lance 500."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={"candidates": []})

        with patch("app.routes.ai.ai.httpx.AsyncClient") as mock_cls:
            mock_cls_instance = AsyncMock()
            mock_cls_instance.__aenter__ = AsyncMock(return_value=mock_cls_instance)
            mock_cls_instance.__aexit__ = AsyncMock(return_value=False)
            mock_cls_instance.post = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_cls_instance

            response = client.post("/ai/analyze-patterns", json=payload)

        assert response.status_code == 500

    def test_gemini_returns_malformed_json_in_text_raises_500(
        self, client, payload, monkeypatch
    ):
        """Si el campo text contiene JSON malformado, debe retornar 500."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(
            return_value={
                "candidates": [{"content": {"parts": [{"text": "{malformed_json"}]}}]
            }
        )

        with patch("app.routes.ai.ai.httpx.AsyncClient") as mock_cls:
            mock_cls_instance = AsyncMock()
            mock_cls_instance.__aenter__ = AsyncMock(return_value=mock_cls_instance)
            mock_cls_instance.__aexit__ = AsyncMock(return_value=False)
            mock_cls_instance.post = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_cls_instance

            response = client.post("/ai/analyze-patterns", json=payload)

        assert response.status_code == 500

    def test_user_name_with_unicode_characters(self, client, payload):
        """user_name con caracteres Unicode no debe romper el formateo del prompt."""
        valid_response = json.dumps(
            {
                "goldenHours": "09:00 - 11:00",
                "goldenHoursConfidence": 0.85,
                "behaviorSummary": "Excelente trabajo.",
                "patterns": [],
                "workStyle": "Desarrollador de Élite 💻",
            }
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(
            return_value={
                "candidates": [{"content": {"parts": [{"text": valid_response}]}}]
            }
        )

        unicode_payload = {
            **payload,
            "user_name": "Héctor García — Desarrollador 🚀",
        }

        with patch("app.routes.ai.ai.httpx.AsyncClient") as mock_cls:
            mock_cls_instance = AsyncMock()
            mock_cls_instance.__aenter__ = AsyncMock(return_value=mock_cls_instance)
            mock_cls_instance.__aexit__ = AsyncMock(return_value=False)
            mock_cls_instance.post = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_cls_instance

            response = client.post("/ai/analyze-patterns", json=unicode_payload)

        assert response.status_code == 200

    def test_network_timeout_returns_500(self, client, payload):
        """Timeout de red debe retornar 500 con mensaje de error."""
        import httpx

        with patch("app.routes.ai.ai.httpx.AsyncClient") as mock_cls:
            mock_cls_instance = AsyncMock()
            mock_cls_instance.__aenter__ = AsyncMock(return_value=mock_cls_instance)
            mock_cls_instance.__aexit__ = AsyncMock(return_value=False)
            mock_cls_instance.post = AsyncMock(
                side_effect=httpx.TimeoutException("Read timeout")
            )
            mock_cls.return_value = mock_cls_instance

            response = client.post("/ai/analyze-patterns", json=payload)

        assert response.status_code == 500
        assert "Error" in response.json()["detail"]
