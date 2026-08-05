"""
tests/conftest.py

Fixtures globales: TestClient de FastAPI, mocks de settings y
factorías de datos reutilizables en toda la suite.
"""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from app.main import app


# ─── Client ────────────────────────────────────────────────────────────────────


@pytest.fixture
def client() -> TestClient:
    """Cliente HTTP síncrono para tests de routes."""
    return TestClient(app, raise_server_exceptions=True)


# ─── Settings mocks ────────────────────────────────────────────────────────────


@pytest.fixture
def gemini_settings(monkeypatch):
    """Simula que GOOGLE_GENERATIVE_AI_API_KEY está configurada.

    Parcheamos tanto app.config.settings como el settings en los módulos de route,
    ya que cada módulo importa `settings` al inicio y tiene su propia referencia.
    """
    monkeypatch.setattr(
        "app.config.settings.GOOGLE_GENERATIVE_AI_API_KEY", "fake-gemini-key"
    )
    monkeypatch.setattr("app.config.settings.ANTHROPIC_API_KEY", "")
    # Parchear en el módulo de routes directamente para que el handler lea el valor correcto
    monkeypatch.setattr("app.routes.ai.ai.settings.ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(
        "app.routes.ai.ai.settings.GOOGLE_GENERATIVE_AI_API_KEY", "fake-gemini-key"
    )
    return "fake-gemini-key"


@pytest.fixture
def anthropic_settings(monkeypatch):
    """Simula que ANTHROPIC_API_KEY está configurada (activa la rama Claude)."""
    monkeypatch.setattr("app.config.settings.ANTHROPIC_API_KEY", "fake-anthropic-key")
    monkeypatch.setattr("app.config.settings.GOOGLE_GENERATIVE_AI_API_KEY", "")
    monkeypatch.setattr(
        "app.routes.ai.ai.settings.ANTHROPIC_API_KEY", "fake-anthropic-key"
    )
    monkeypatch.setattr("app.routes.ai.ai.settings.GOOGLE_GENERATIVE_AI_API_KEY", "")
    return "fake-anthropic-key"


@pytest.fixture
def no_api_keys(monkeypatch):
    """Simula un entorno sin ninguna API key configurada."""
    monkeypatch.setattr("app.config.settings.GOOGLE_GENERATIVE_AI_API_KEY", "")
    monkeypatch.setattr("app.config.settings.ANTHROPIC_API_KEY", "")
    monkeypatch.setattr("app.routes.ai.ai.settings.GOOGLE_GENERATIVE_AI_API_KEY", "")
    monkeypatch.setattr("app.routes.ai.ai.settings.ANTHROPIC_API_KEY", "")


# ─── Data factories ────────────────────────────────────────────────────────────


@pytest.fixture
def sample_messages():
    return [
        {"role": "user", "content": "¿Cómo organizo mi semana?"},
    ]


@pytest.fixture
def sample_messages_with_history():
    return [
        {"role": "user", "content": "Hola, quiero ser más productivo"},
        {"role": "assistant", "content": "Claro, te ayudo."},
        {"role": "user", "content": "¿Cómo organizo mis tareas?"},
    ]


@pytest.fixture
def sample_tasks():
    return [
        {
            "id": "task-001",
            "title": "Implementar autenticación JWT",
            "description": "Agregar login con tokens JWT al backend",
            "priority": "HIGH",
            "deadline": "2026-08-10",
            "status": "todo",
            "estimatedTime": "3h",
        },
        {
            "id": "task-002",
            "title": "Revisar PRs pendientes",
            "description": None,
            "priority": "MEDIUM",
            "deadline": None,
            "status": "in_progress",
            "estimatedTime": "1h",
        },
        {
            "id": "task-003",
            "title": "Actualizar documentación API",
            "description": "Agregar ejemplos de uso a todos los endpoints",
            "priority": "LOW",
            "deadline": "2026-08-20",
            "status": "todo",
            "estimatedTime": None,
        },
    ]


@pytest.fixture
def sample_free_slots():
    return [
        {"start": "2026-08-05T09:00:00", "end": "2026-08-05T11:00:00"},
        {"start": "2026-08-05T14:00:00", "end": "2026-08-05T17:00:00"},
    ]


@pytest.fixture
def sample_patterns_payload():
    return {
        "user_name": "Alexis",
        "hour_buckets": {
            "9": {"sessions": 3, "tasks_completed": 5, "focus_minutes": 120},
            "14": {"sessions": 2, "tasks_completed": 3, "focus_minutes": 80},
            "22": {"sessions": 4, "tasks_completed": 7, "focus_minutes": 200},
        },
        "task_stats": {
            "total_completed": 15,
            "completion_rate": 0.87,
            "avg_time_per_task": 42,
        },
        "session_stats": {
            "total_sessions": 9,
            "avg_duration_minutes": 45,
            "longest_streak_days": 7,
        },
        "top_productive_hours": [22, 9, 14],
        "work_style_hint": "night_owl",
    }


# ─── Gemini response mock helpers ──────────────────────────────────────────────


def make_gemini_organize_response(tasks: list[dict]) -> str:
    """Genera JSON válido de respuesta de organización de tareas."""
    import json

    plan = [
        {
            "taskId": t["id"],
            "recommendedPriority": t.get("priority", "MEDIUM"),
            "suggestedOrder": i + 1,
            "reason": f"Tarea {t['title']} priorizada por deadline y urgencia.",
            "suggestedDate": t.get("deadline"),
            "estimatedTime": t.get("estimatedTime"),
        }
        for i, t in enumerate(tasks)
    ]
    return json.dumps({"plan": plan})


def make_gemini_calendar_response() -> str:
    import json

    return json.dumps(
        {
            "events": [
                {
                    "taskId": "task-001",
                    "title": "Implementar autenticación JWT",
                    "startTime": "2026-08-05T09:00:00",
                    "endTime": "2026-08-05T11:00:00",
                    "reason": "Alta prioridad, slot disponible en la mañana.",
                },
            ]
        }
    )


def make_gemini_weekly_response() -> str:
    import json

    return json.dumps(
        {
            "weeklyPlan": [
                {"day": "Monday", "tasks": ["Implementar autenticación JWT"]},
                {"day": "Tuesday", "tasks": ["Revisar PRs pendientes"]},
                {"day": "Wednesday", "tasks": []},
                {"day": "Thursday", "tasks": ["Actualizar documentación API"]},
                {"day": "Friday", "tasks": []},
                {"day": "Saturday", "tasks": []},
                {"day": "Sunday", "tasks": []},
            ],
            "recommendationSummary": "Semana equilibrada con tareas de alta prioridad al inicio.",
        }
    )


def make_gemini_mock_client(response_text: str) -> MagicMock:
    """Crea un mock del cliente Gemini con .models.generate_content."""
    mock_response = MagicMock()
    mock_response.text = response_text

    mock_models = MagicMock()
    mock_models.generate_content.return_value = mock_response

    mock_client = MagicMock()
    mock_client.models = mock_models
    return mock_client
