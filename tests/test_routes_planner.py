"""
tests/test_routes_planner.py

Tests funcionales del router /ai/planner (organize, calendar, weekly, improve).
Todos los tests mockean el cliente Gemini SDK para aislar la lógica del router
de las llamadas reales a la API.
"""

import json
from unittest.mock import patch
from fastapi.testclient import TestClient

from tests.conftest import (
    make_gemini_mock_client,
    make_gemini_organize_response,
    make_gemini_calendar_response,
    make_gemini_weekly_response,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. POST /ai/planner/organize
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrganizeTasksValidation:
    """Valida que el endpoint rechace payloads inválidos."""

    def test_missing_tasks_field_returns_422(self, client: TestClient):
        response = client.post("/ai/planner/organize", json={})
        assert response.status_code == 422

    def test_tasks_must_be_a_list(self, client: TestClient):
        response = client.post("/ai/planner/organize", json={"tasks": "not-a-list"})
        assert response.status_code == 422

    def test_empty_tasks_list_is_valid_input(self, client: TestClient, sample_tasks):
        """Lista vacía es un caso real: usuario sin tareas."""
        gemini_response = json.dumps({"plan": []})
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post("/ai/planner/organize", json={"tasks": []})
        assert response.status_code == 200
        assert response.json()["plan"] == []


class TestOrganizeTasksSuccess:
    """Valida el flujo exitoso de organización de tareas."""

    def test_returns_plan_with_all_task_ids(self, client: TestClient, sample_tasks):
        gemini_response = make_gemini_organize_response(sample_tasks)
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post("/ai/planner/organize", json={"tasks": sample_tasks})

        assert response.status_code == 200
        plan = response.json()["plan"]
        returned_ids = {item["taskId"] for item in plan}
        expected_ids = {t["id"] for t in sample_tasks}
        assert returned_ids == expected_ids

    def test_plan_items_have_required_fields(self, client: TestClient, sample_tasks):
        gemini_response = make_gemini_organize_response(sample_tasks)
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post("/ai/planner/organize", json={"tasks": sample_tasks})

        for item in response.json()["plan"]:
            assert "taskId" in item
            assert "recommendedPriority" in item
            assert "suggestedOrder" in item
            assert "reason" in item

    def test_priority_values_are_valid(self, client: TestClient, sample_tasks):
        gemini_response = make_gemini_organize_response(sample_tasks)
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post("/ai/planner/organize", json={"tasks": sample_tasks})

        valid_priorities = {"HIGH", "MEDIUM", "LOW"}
        for item in response.json()["plan"]:
            assert item["recommendedPriority"] in valid_priorities

    def test_suggested_order_is_sequential(self, client: TestClient, sample_tasks):
        gemini_response = make_gemini_organize_response(sample_tasks)
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post("/ai/planner/organize", json={"tasks": sample_tasks})

        orders = sorted(item["suggestedOrder"] for item in response.json()["plan"])
        assert orders == list(range(1, len(sample_tasks) + 1))

    def test_task_without_description_is_handled(self, client: TestClient):
        """Tarea sin descripción ni deadline son casos reales de producción."""
        tasks = [{"id": "t1", "title": "Sin descripción"}]
        gemini_response = json.dumps(
            {
                "plan": [
                    {
                        "taskId": "t1",
                        "recommendedPriority": "LOW",
                        "suggestedOrder": 1,
                        "reason": "ok",
                    }
                ]
            }
        )
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post("/ai/planner/organize", json={"tasks": tasks})

        assert response.status_code == 200

    def test_gemini_sdk_called_with_json_mime_type(
        self, client: TestClient, sample_tasks
    ):
        gemini_response = make_gemini_organize_response(sample_tasks)
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            client.post("/ai/planner/organize", json={"tasks": sample_tasks})

        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert call_kwargs["config"].response_mime_type == "application/json"

    def test_gemini_sdk_called_with_correct_model(
        self, client: TestClient, sample_tasks
    ):
        gemini_response = make_gemini_organize_response(sample_tasks)
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            client.post("/ai/planner/organize", json={"tasks": sample_tasks})

        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert call_kwargs["model"] == "gemini-2.5-flash"

    def test_large_task_list_is_handled(self, client: TestClient):
        """Simula un usuario con 50 tareas — caso real de power user."""
        tasks = [
            {"id": f"task-{i}", "title": f"Tarea {i}", "priority": "MEDIUM"}
            for i in range(50)
        ]
        plan_items = [
            {
                "taskId": t["id"],
                "recommendedPriority": "MEDIUM",
                "suggestedOrder": i + 1,
                "reason": "ok",
            }
            for i, t in enumerate(tasks)
        ]
        gemini_response = json.dumps({"plan": plan_items})
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post("/ai/planner/organize", json={"tasks": tasks})

        assert response.status_code == 200
        assert len(response.json()["plan"]) == 50


# ═══════════════════════════════════════════════════════════════════════════════
# 2. POST /ai/planner/calendar
# ═══════════════════════════════════════════════════════════════════════════════


class TestCalendarPlannerValidation:
    def test_missing_tasks_returns_422(self, client: TestClient):
        response = client.post(
            "/ai/planner/calendar",
            json={
                "free_slots": [
                    {"start": "2026-08-05T09:00:00", "end": "2026-08-05T11:00:00"}
                ]
            },
        )
        assert response.status_code == 422

    def test_missing_free_slots_returns_422(self, client: TestClient, sample_tasks):
        response = client.post("/ai/planner/calendar", json={"tasks": sample_tasks})
        assert response.status_code == 422

    def test_empty_slots_is_valid(self, client: TestClient, sample_tasks):
        """Sin slots disponibles, el resultado debe ser una lista de eventos vacía."""
        gemini_response = json.dumps({"events": []})
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post(
                "/ai/planner/calendar",
                json={"tasks": sample_tasks, "free_slots": []},
            )
        assert response.status_code == 200
        assert response.json()["events"] == []


class TestCalendarPlannerSuccess:
    def test_events_have_required_fields(
        self, client: TestClient, sample_tasks, sample_free_slots
    ):
        gemini_response = make_gemini_calendar_response()
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post(
                "/ai/planner/calendar",
                json={"tasks": sample_tasks, "free_slots": sample_free_slots},
            )

        for event in response.json()["events"]:
            assert "title" in event
            assert "startTime" in event
            assert "endTime" in event
            assert "reason" in event

    def test_start_time_before_end_time(
        self, client: TestClient, sample_tasks, sample_free_slots
    ):
        gemini_response = make_gemini_calendar_response()
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post(
                "/ai/planner/calendar",
                json={"tasks": sample_tasks, "free_slots": sample_free_slots},
            )

        for event in response.json()["events"]:
            assert event["startTime"] < event["endTime"], (
                f"startTime {event['startTime']} >= endTime {event['endTime']}"
            )

    def test_task_id_in_event_can_be_null(
        self, client: TestClient, sample_tasks, sample_free_slots
    ):
        """taskId es nullable — eventos de bloqueo (buffer) no tienen tarea."""
        gemini_response = json.dumps(
            {
                "events": [
                    {
                        "taskId": None,
                        "title": "Buffer de descanso",
                        "startTime": "2026-08-05T11:00:00",
                        "endTime": "2026-08-05T11:30:00",
                        "reason": "Tiempo de recuperación",
                    }
                ]
            }
        )
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post(
                "/ai/planner/calendar",
                json={"tasks": sample_tasks, "free_slots": sample_free_slots},
            )

        assert response.status_code == 200
        assert response.json()["events"][0]["taskId"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. POST /ai/planner/weekly
# ═══════════════════════════════════════════════════════════════════════════════


class TestWeeklyPlannerValidation:
    def test_missing_tasks_returns_422(self, client: TestClient):
        response = client.post("/ai/planner/weekly", json={})
        assert response.status_code == 422

    def test_availability_defaults_when_not_provided(
        self, client: TestClient, sample_tasks
    ):
        """availability es opcional; cuando falta, el router usa un default."""
        gemini_response = make_gemini_weekly_response()
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post("/ai/planner/weekly", json={"tasks": sample_tasks})
        assert response.status_code == 200


class TestWeeklyPlannerSuccess:
    def test_weekly_plan_has_seven_days(self, client: TestClient, sample_tasks):
        gemini_response = make_gemini_weekly_response()
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post("/ai/planner/weekly", json={"tasks": sample_tasks})

        assert response.status_code == 200
        assert len(response.json()["weeklyPlan"]) == 7

    def test_each_day_has_day_name_and_tasks(self, client: TestClient, sample_tasks):
        gemini_response = make_gemini_weekly_response()
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post("/ai/planner/weekly", json={"tasks": sample_tasks})

        for day_plan in response.json()["weeklyPlan"]:
            assert "day" in day_plan
            assert "tasks" in day_plan
            assert isinstance(day_plan["tasks"], list)

    def test_recommendation_summary_is_present(self, client: TestClient, sample_tasks):
        gemini_response = make_gemini_weekly_response()
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post("/ai/planner/weekly", json={"tasks": sample_tasks})

        assert "recommendationSummary" in response.json()
        assert len(response.json()["recommendationSummary"]) > 0

    def test_day_names_are_valid_english(self, client: TestClient, sample_tasks):
        valid_days = {
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        }
        gemini_response = make_gemini_weekly_response()
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post("/ai/planner/weekly", json={"tasks": sample_tasks})

        for day_plan in response.json()["weeklyPlan"]:
            assert day_plan["day"] in valid_days

    def test_custom_availability_is_passed_to_planner(
        self, client: TestClient, sample_tasks
    ):
        """La disponibilidad personalizada debe llegar al modelo."""
        gemini_response = make_gemini_weekly_response()
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post(
                "/ai/planner/weekly",
                json={
                    "tasks": sample_tasks,
                    "availability": {
                        "working_hours": "08:00 - 12:00",
                        "days_off": ["Sunday"],
                    },
                },
            )

        assert response.status_code == 200
        # Verificar que generate_content fue llamado con availability en el prompt
        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert "08:00 - 12:00" in call_kwargs["contents"]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. POST /ai/planner/improve
# ═══════════════════════════════════════════════════════════════════════════════


class TestTaskImproveValidation:
    def test_missing_title_returns_422(self, client: TestClient):
        response = client.post(
            "/ai/planner/improve",
            json={"description": "sin título", "mode": "subtasks"},
        )
        assert response.status_code == 422

    def test_missing_mode_returns_422(self, client: TestClient):
        response = client.post(
            "/ai/planner/improve",
            json={"title": "Tarea test"},
        )
        assert response.status_code == 422

    def test_description_is_optional(self, client: TestClient):
        gemini_response = json.dumps({"subtasks": ["Paso 1", "Paso 2"]})
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post(
                "/ai/planner/improve",
                json={"title": "Implementar login", "mode": "subtasks"},
            )
        assert response.status_code == 200

    def test_null_description_is_handled(self, client: TestClient):
        gemini_response = json.dumps({"subtasks": ["Paso 1"]})
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post(
                "/ai/planner/improve",
                json={"title": "Tarea", "description": None, "mode": "subtasks"},
            )
        assert response.status_code == 200


class TestTaskImproveSubtasks:
    def test_subtasks_mode_returns_list(self, client: TestClient):
        gemini_response = json.dumps(
            {"subtasks": ["Diseñar schema", "Implementar endpoints", "Escribir tests"]}
        )
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post(
                "/ai/planner/improve",
                json={
                    "title": "Crear API REST",
                    "description": "CRUD completo",
                    "mode": "subtasks",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "subtasks" in data
        assert isinstance(data["subtasks"], list)
        assert len(data["subtasks"]) > 0

    def test_subtasks_are_non_empty_strings(self, client: TestClient):
        gemini_response = json.dumps({"subtasks": ["Paso A", "Paso B"]})
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post(
                "/ai/planner/improve",
                json={"title": "Tarea", "mode": "subtasks"},
            )

        for subtask in response.json()["subtasks"]:
            assert isinstance(subtask, str)
            assert len(subtask.strip()) > 0


class TestTaskImproveEstimate:
    def test_estimate_mode_returns_duration_string(self, client: TestClient):
        gemini_response = json.dumps({"estimatedTime": "2h 30m"})
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post(
                "/ai/planner/improve",
                json={"title": "Implementar autenticación", "mode": "estimate"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "estimatedTime" in data
        assert isinstance(data["estimatedTime"], str)

    def test_estimate_format_contains_time_unit(self, client: TestClient):
        """El tiempo estimado debe contener alguna unidad (h, m, min, hora)."""
        gemini_response = json.dumps({"estimatedTime": "1h 45m"})
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post(
                "/ai/planner/improve",
                json={"title": "Tarea", "mode": "estimate"},
            )

        estimated = response.json()["estimatedTime"].lower()
        has_time_unit = any(
            unit in estimated for unit in ["h", "m", "min", "hora", "minute"]
        )
        assert has_time_unit


class TestTaskImprovePriority:
    def test_priority_mode_returns_valid_priority(self, client: TestClient):
        for priority in ["HIGH", "MEDIUM", "LOW"]:
            gemini_response = json.dumps({"suggestedPriority": priority})
            mock_client = make_gemini_mock_client(gemini_response)

            with patch(
                "app.services.ai.planner.genai.Client", return_value=mock_client
            ):
                response = client.post(
                    "/ai/planner/improve",
                    json={"title": "Tarea urgente", "mode": "priority"},
                )

            assert response.status_code == 200
            assert response.json()["suggestedPriority"] == priority


class TestTaskImproveAll:
    def test_all_mode_returns_complete_response(self, client: TestClient):
        gemini_response = json.dumps(
            {
                "subtasks": ["Diseñar", "Implementar", "Testear"],
                "estimatedTime": "4h",
                "suggestedPriority": "HIGH",
            }
        )
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post(
                "/ai/planner/improve",
                json={
                    "title": "Feature completa",
                    "description": "Módulo de auth",
                    "mode": "all",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "subtasks" in data
        assert "estimatedTime" in data
        assert "suggestedPriority" in data

    def test_all_mode_subtasks_is_a_list(self, client: TestClient):
        gemini_response = json.dumps(
            {
                "subtasks": ["Paso 1", "Paso 2"],
                "estimatedTime": "2h",
                "suggestedPriority": "MEDIUM",
            }
        )
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post(
                "/ai/planner/improve",
                json={"title": "Tarea", "mode": "all"},
            )

        assert isinstance(response.json()["subtasks"], list)

    def test_unknown_mode_falls_through_to_all(self, client: TestClient):
        """Modo desconocido cae en el else (mode == 'all') por diseño."""
        gemini_response = json.dumps(
            {
                "subtasks": ["Step 1"],
                "estimatedTime": "1h",
                "suggestedPriority": "LOW",
            }
        )
        mock_client = make_gemini_mock_client(gemini_response)

        with patch("app.services.ai.planner.genai.Client", return_value=mock_client):
            response = client.post(
                "/ai/planner/improve",
                json={"title": "Tarea", "mode": "unknown_mode"},
            )

        # El servicio trata cualquier modo no reconocido como "all"
        assert response.status_code == 200
