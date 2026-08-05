"""
tests/test_service_planner.py

Tests UNITARIOS DIRECTOS de AIPlannerService.
Estrategia: mock de genai.Client a nivel de módulo para aislar completamente
el servicio de red. Se prueba la lógica interna: construcción de prompts,
parsing de respuestas, manejo de ValidationError y edge cases de producción.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from pydantic import ValidationError

from app.services.ai.planner import (
    AIPlannerService,
    TaskPlanItem,
    TasksOrganizeResponse,
    TimeBlockItem,
    CalendarPlannerResponse,
    WeeklyPlanDayItem,
    WeeklyPlanResponse,
    SubtasksResponse,
    EstimatedTimeResponse,
    SuggestedPriorityResponse,
    ImproveAllResponse,
)


# ─── Factory helpers ────────────────────────────────────────────────────────────


def make_service(response_text: str) -> AIPlannerService:
    """Crea AIPlannerService con cliente Gemini completamente mockeado."""
    mock_response = MagicMock()
    mock_response.text = response_text

    mock_models = MagicMock()
    mock_models.generate_content.return_value = mock_response

    mock_client = MagicMock()
    mock_client.models = mock_models

    with (
        patch("app.services.ai.planner.settings") as mock_settings,
        patch("app.services.ai.planner.genai.Client", return_value=mock_client),
    ):
        mock_settings.GOOGLE_GENERATIVE_AI_API_KEY = "test-key"
        service = AIPlannerService()

    service.client = mock_client
    return service


def make_tasks(n: int = 3) -> list[dict]:
    return [
        {
            "id": f"task-{i:03d}",
            "title": f"Tarea de prueba {i}",
            "description": f"Descripción de la tarea {i}",
            "priority": "HIGH" if i == 1 else "MEDIUM",
            "deadline": f"2026-08-{10 + i:02d}",
            "status": "todo",
            "estimatedTime": "1h",
        }
        for i in range(1, n + 1)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PYDANTIC MODELS — Tests unitarios de schemas
# ═══════════════════════════════════════════════════════════════════════════════


class TestTaskPlanItemModel:
    """Valida el schema de TaskPlanItem con casos realistas."""

    def test_valid_task_plan_item(self):
        item = TaskPlanItem(
            taskId="task-001",
            recommendedPriority="HIGH",
            suggestedOrder=1,
            reason="Deadline más cercano.",
        )
        assert item.taskId == "task-001"
        assert item.suggestedDate is None
        assert item.estimatedTime is None

    def test_all_fields_present(self):
        item = TaskPlanItem(
            taskId="t1",
            recommendedPriority="MEDIUM",
            suggestedOrder=2,
            reason="Impacto moderado.",
            suggestedDate="2026-08-10",
            estimatedTime="2h 30m",
        )
        assert item.suggestedDate == "2026-08-10"
        assert item.estimatedTime == "2h 30m"

    def test_missing_required_task_id_raises(self):
        with pytest.raises(ValidationError):
            TaskPlanItem(recommendedPriority="HIGH", suggestedOrder=1, reason="ok")

    def test_missing_reason_raises(self):
        with pytest.raises(ValidationError):
            TaskPlanItem(taskId="t1", recommendedPriority="HIGH", suggestedOrder=1)

    def test_suggested_order_must_be_int(self):
        with pytest.raises(ValidationError):
            TaskPlanItem(
                taskId="t1",
                recommendedPriority="HIGH",
                suggestedOrder="primero",
                reason="ok",
            )

    def test_null_suggested_date_accepted(self):
        item = TaskPlanItem(
            taskId="t1",
            recommendedPriority="LOW",
            suggestedOrder=3,
            reason="Sin urgencia.",
            suggestedDate=None,
        )
        assert item.suggestedDate is None

    def test_unicode_in_reason(self):
        item = TaskPlanItem(
            taskId="t1",
            recommendedPriority="HIGH",
            suggestedOrder=1,
            reason="Urgente: presentación con cliente 🚀 — mañana al mediodía.",
        )
        assert "🚀" in item.reason


class TestTasksOrganizeResponseModel:
    def test_empty_plan_is_valid(self):
        r = TasksOrganizeResponse(plan=[])
        assert r.plan == []

    def test_plan_with_items(self):
        items = [
            TaskPlanItem(
                taskId=f"t{i}",
                recommendedPriority="MEDIUM",
                suggestedOrder=i,
                reason="ok",
            )
            for i in range(1, 4)
        ]
        r = TasksOrganizeResponse(plan=items)
        assert len(r.plan) == 3

    def test_model_validate_json_valid(self):
        raw = json.dumps(
            {
                "plan": [
                    {
                        "taskId": "t1",
                        "recommendedPriority": "HIGH",
                        "suggestedOrder": 1,
                        "reason": "Urgente",
                    }
                ]
            }
        )
        r = TasksOrganizeResponse.model_validate_json(raw)
        assert r.plan[0].taskId == "t1"

    def test_model_validate_json_invalid_raises(self):
        with pytest.raises((ValidationError, Exception)):
            TasksOrganizeResponse.model_validate_json("{malformed}")

    def test_model_validate_json_missing_plan_raises(self):
        with pytest.raises(ValidationError):
            TasksOrganizeResponse.model_validate_json('{"other": "field"}')


class TestTimeBlockItemModel:
    def test_valid_time_block(self):
        item = TimeBlockItem(
            taskId="t1",
            title="Implementar login",
            startTime="2026-08-05T09:00:00",
            endTime="2026-08-05T11:00:00",
            reason="Slot disponible en la mañana.",
        )
        assert item.taskId == "t1"

    def test_null_task_id_accepted(self):
        item = TimeBlockItem(
            taskId=None,
            title="Buffer de descanso",
            startTime="2026-08-05T11:00:00",
            endTime="2026-08-05T11:30:00",
            reason="Tiempo de recuperación.",
        )
        assert item.taskId is None

    def test_missing_title_raises(self):
        with pytest.raises(ValidationError):
            TimeBlockItem(
                startTime="2026-08-05T09:00:00",
                endTime="2026-08-05T10:00:00",
                reason="ok",
            )


class TestWeeklyPlanModels:
    def test_weekly_plan_day_item_valid(self):
        day = WeeklyPlanDayItem(day="Monday", tasks=["Tarea 1", "Tarea 2"])
        assert day.day == "Monday"
        assert len(day.tasks) == 2

    def test_weekly_plan_day_empty_tasks(self):
        day = WeeklyPlanDayItem(day="Sunday", tasks=[])
        assert day.tasks == []

    def test_weekly_plan_response_validate_json(self):
        raw = json.dumps(
            {
                "weeklyPlan": [
                    {"day": "Monday", "tasks": ["Tarea A"]},
                    {"day": "Tuesday", "tasks": []},
                    {"day": "Wednesday", "tasks": ["Tarea B"]},
                    {"day": "Thursday", "tasks": []},
                    {"day": "Friday", "tasks": ["Tarea C"]},
                    {"day": "Saturday", "tasks": []},
                    {"day": "Sunday", "tasks": []},
                ],
                "recommendationSummary": "Semana balanceada.",
            }
        )
        r = WeeklyPlanResponse.model_validate_json(raw)
        assert len(r.weeklyPlan) == 7
        assert r.recommendationSummary == "Semana balanceada."


class TestImproveModels:
    def test_subtasks_response_valid(self):
        r = SubtasksResponse(subtasks=["Paso 1", "Paso 2", "Paso 3"])
        assert len(r.subtasks) == 3

    def test_estimated_time_response_valid(self):
        r = EstimatedTimeResponse(estimatedTime="2h 30m")
        assert r.estimatedTime == "2h 30m"

    def test_suggested_priority_response_valid(self):
        r = SuggestedPriorityResponse(suggestedPriority="HIGH")
        assert r.suggestedPriority == "HIGH"

    def test_improve_all_response_valid(self):
        r = ImproveAllResponse(
            subtasks=["Diseñar", "Implementar"],
            estimatedTime="3h",
            suggestedPriority="MEDIUM",
        )
        assert len(r.subtasks) == 2

    def test_improve_all_missing_field_raises(self):
        with pytest.raises(ValidationError):
            ImproveAllResponse(subtasks=["ok"], estimatedTime="1h")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. organize_tasks() — Tests unitarios del método
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrganizeTasksService:
    """Prueba directamente AIPlannerService.organize_tasks()."""

    def _make_response(self, tasks: list[dict]) -> str:
        plan = [
            {
                "taskId": t["id"],
                "recommendedPriority": t.get("priority", "MEDIUM"),
                "suggestedOrder": i + 1,
                "reason": "Priorizado por deadline.",
            }
            for i, t in enumerate(tasks)
        ]
        return json.dumps({"plan": plan})

    @pytest.mark.asyncio
    async def test_organize_tasks_calls_gemini(self):
        tasks = make_tasks(2)
        service = make_service(self._make_response(tasks))
        result = await service.organize_tasks(tasks)
        assert service.client.models.generate_content.called

    @pytest.mark.asyncio
    async def test_organize_tasks_returns_organize_response_type(self):
        tasks = make_tasks(3)
        service = make_service(self._make_response(tasks))
        result = await service.organize_tasks(tasks)
        assert isinstance(result, TasksOrganizeResponse)

    @pytest.mark.asyncio
    async def test_organize_tasks_prompt_contains_task_ids(self):
        tasks = make_tasks(3)
        service = make_service(self._make_response(tasks))
        await service.organize_tasks(tasks)
        call_args = service.client.models.generate_content.call_args
        prompt = call_args[1]["contents"]
        for t in tasks:
            assert t["id"] in prompt, f"Task ID {t['id']} not in prompt"

    @pytest.mark.asyncio
    async def test_organize_tasks_prompt_contains_task_titles(self):
        tasks = make_tasks(3)
        service = make_service(self._make_response(tasks))
        await service.organize_tasks(tasks)
        prompt = service.client.models.generate_content.call_args[1]["contents"]
        for t in tasks:
            assert t["title"] in prompt

    @pytest.mark.asyncio
    async def test_organize_tasks_uses_json_mime_type(self):
        tasks = make_tasks(2)
        service = make_service(self._make_response(tasks))
        await service.organize_tasks(tasks)
        call_args = service.client.models.generate_content.call_args[1]
        assert call_args["config"].response_mime_type == "application/json"

    @pytest.mark.asyncio
    async def test_organize_tasks_uses_correct_model(self):
        tasks = make_tasks(2)
        service = make_service(self._make_response(tasks))
        await service.organize_tasks(tasks)
        call_args = service.client.models.generate_content.call_args[1]
        assert call_args["model"] == "gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_organize_tasks_empty_list(self):
        service = make_service(json.dumps({"plan": []}))
        result = await service.organize_tasks([])
        assert result.plan == []

    @pytest.mark.asyncio
    async def test_organize_tasks_task_without_description(self):
        tasks = [{"id": "t1", "title": "Sin descripción"}]
        service = make_service(
            json.dumps(
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
        )
        result = await service.organize_tasks(tasks)
        assert result.plan[0].taskId == "t1"

    @pytest.mark.asyncio
    async def test_organize_tasks_task_with_none_fields(self):
        """Caso real: tarea creada rápido sin deadline ni prioridad."""
        tasks = [
            {
                "id": "t1",
                "title": "Tarea incompleta",
                "description": None,
                "priority": None,
                "deadline": None,
                "status": None,
                "estimatedTime": None,
            }
        ]
        service = make_service(
            json.dumps(
                {
                    "plan": [
                        {
                            "taskId": "t1",
                            "recommendedPriority": "LOW",
                            "suggestedOrder": 1,
                            "reason": "Sin deadline, baja prioridad.",
                        }
                    ]
                }
            )
        )
        result = await service.organize_tasks(tasks)
        assert result.plan[0].taskId == "t1"

    @pytest.mark.asyncio
    async def test_organize_tasks_task_with_unicode_title(self):
        """Caracteres Unicode en título no deben romper el prompt."""
        tasks = [
            {
                "id": "t1",
                "title": "Implementar autenticación — módulo de IA 🚀",
                "description": "Con soporte para OAuth2 y PKCE",
            }
        ]
        service = make_service(
            json.dumps(
                {
                    "plan": [
                        {
                            "taskId": "t1",
                            "recommendedPriority": "HIGH",
                            "suggestedOrder": 1,
                            "reason": "Alta importancia.",
                        }
                    ]
                }
            )
        )
        result = await service.organize_tasks(tasks)
        prompt = service.client.models.generate_content.call_args[1]["contents"]
        assert "🚀" in prompt
        assert "autenticación" in prompt

    @pytest.mark.asyncio
    async def test_organize_tasks_invalid_gemini_json_raises(self):
        """Si Gemini retorna JSON inválido, model_validate_json debe lanzar excepción."""
        service = make_service("invalid json response")
        with pytest.raises(Exception):
            await service.organize_tasks(make_tasks(1))

    @pytest.mark.asyncio
    async def test_organize_tasks_50_tasks(self):
        """Caso de poder usuario con muchas tareas."""
        tasks = [
            {"id": f"task-{i}", "title": f"Tarea {i}", "priority": "MEDIUM"}
            for i in range(50)
        ]
        plan_items = [
            {
                "taskId": f"task-{i}",
                "recommendedPriority": "MEDIUM",
                "suggestedOrder": i + 1,
                "reason": "ok",
            }
            for i in range(50)
        ]
        service = make_service(json.dumps({"plan": plan_items}))
        result = await service.organize_tasks(tasks)
        assert len(result.plan) == 50

    @pytest.mark.asyncio
    async def test_organize_tasks_no_description_uses_placeholder(self):
        """Tarea sin descripción debe usar 'No description' en el prompt."""
        tasks = [{"id": "t1", "title": "Sin desc", "description": None}]
        service = make_service(
            json.dumps(
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
        )
        await service.organize_tasks(tasks)
        prompt = service.client.models.generate_content.call_args[1]["contents"]
        assert "No description" in prompt


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ai_calendar_planner() — Tests unitarios
# ═══════════════════════════════════════════════════════════════════════════════


class TestCalendarPlannerService:
    def _make_calendar_response(self) -> str:
        return json.dumps(
            {
                "events": [
                    {
                        "taskId": "task-001",
                        "title": "Implementar login",
                        "startTime": "2026-08-05T09:00:00",
                        "endTime": "2026-08-05T11:00:00",
                        "reason": "Slot disponible.",
                    }
                ]
            }
        )

    @pytest.mark.asyncio
    async def test_calendar_planner_returns_calendar_response_type(self):
        service = make_service(self._make_calendar_response())
        result = await service.ai_calendar_planner(make_tasks(1), [])
        assert isinstance(result, CalendarPlannerResponse)

    @pytest.mark.asyncio
    async def test_calendar_planner_prompt_contains_task_title(self):
        tasks = make_tasks(2)
        service = make_service(self._make_calendar_response())
        await service.ai_calendar_planner(tasks, [])
        prompt = service.client.models.generate_content.call_args[1]["contents"]
        assert tasks[0]["title"] in prompt
        assert tasks[1]["title"] in prompt

    @pytest.mark.asyncio
    async def test_calendar_planner_prompt_contains_slot_times(self):
        tasks = make_tasks(1)
        slots = [{"start": "2026-08-05T09:00:00", "end": "2026-08-05T12:00:00"}]
        service = make_service(self._make_calendar_response())
        await service.ai_calendar_planner(tasks, slots)
        prompt = service.client.models.generate_content.call_args[1]["contents"]
        assert "2026-08-05T09:00:00" in prompt
        assert "2026-08-05T12:00:00" in prompt

    @pytest.mark.asyncio
    async def test_calendar_planner_empty_slots(self):
        service = make_service(json.dumps({"events": []}))
        result = await service.ai_calendar_planner(make_tasks(3), [])
        assert isinstance(result, CalendarPlannerResponse)
        assert result.events == []

    @pytest.mark.asyncio
    async def test_calendar_planner_task_with_duration(self):
        """Tareas con campo 'duration' explícito deben aparecer en el prompt."""
        tasks = [
            {"id": "t1", "title": "Deep work", "duration": "3h", "priority": "HIGH"}
        ]
        slots = [{"start": "2026-08-05T08:00:00", "end": "2026-08-05T12:00:00"}]
        service = make_service(
            json.dumps(
                {
                    "events": [
                        {
                            "taskId": "t1",
                            "title": "Deep work",
                            "startTime": "2026-08-05T08:00:00",
                            "endTime": "2026-08-05T11:00:00",
                            "reason": "Prioridad alta.",
                        }
                    ]
                }
            )
        )
        await service.ai_calendar_planner(tasks, slots)
        prompt = service.client.models.generate_content.call_args[1]["contents"]
        assert "3h" in prompt

    @pytest.mark.asyncio
    async def test_calendar_planner_default_duration_when_missing(self):
        """Tarea sin 'duration' debe mostrar el valor default '30 mins' en el prompt."""
        tasks = [{"id": "t1", "title": "Quick task"}]
        service = make_service(json.dumps({"events": []}))
        await service.ai_calendar_planner(tasks, [])
        prompt = service.client.models.generate_content.call_args[1]["contents"]
        assert "30 mins" in prompt

    @pytest.mark.asyncio
    async def test_calendar_planner_invalid_gemini_response_raises(self):
        service = make_service("not json at all")
        with pytest.raises(Exception):
            await service.ai_calendar_planner(make_tasks(1), [])

    @pytest.mark.asyncio
    async def test_calendar_planner_multiple_slots(self):
        """Múltiples slots deben aparecer todos en el prompt."""
        tasks = make_tasks(1)
        slots = [
            {"start": "2026-08-05T09:00:00", "end": "2026-08-05T10:00:00"},
            {"start": "2026-08-05T14:00:00", "end": "2026-08-05T16:00:00"},
            {"start": "2026-08-06T09:00:00", "end": "2026-08-06T11:00:00"},
        ]
        service = make_service(json.dumps({"events": []}))
        await service.ai_calendar_planner(tasks, slots)
        prompt = service.client.models.generate_content.call_args[1]["contents"]
        assert "2026-08-05T14:00:00" in prompt
        assert "2026-08-06T09:00:00" in prompt


# ═══════════════════════════════════════════════════════════════════════════════
# 4. weekly_planner() — Tests unitarios
# ═══════════════════════════════════════════════════════════════════════════════


def _weekly_response() -> str:
    return json.dumps(
        {
            "weeklyPlan": [
                {"day": "Monday", "tasks": ["Tarea 1"]},
                {"day": "Tuesday", "tasks": []},
                {"day": "Wednesday", "tasks": ["Tarea 2"]},
                {"day": "Thursday", "tasks": []},
                {"day": "Friday", "tasks": ["Tarea 3"]},
                {"day": "Saturday", "tasks": []},
                {"day": "Sunday", "tasks": []},
            ],
            "recommendationSummary": "Plan equilibrado.",
        }
    )


class TestWeeklyPlannerService:
    @pytest.mark.asyncio
    async def test_weekly_planner_returns_weekly_plan_response(self):
        service = make_service(_weekly_response())
        result = await service.weekly_planner(make_tasks(3), {})
        assert isinstance(result, WeeklyPlanResponse)

    @pytest.mark.asyncio
    async def test_weekly_planner_prompt_contains_task_titles(self):
        tasks = make_tasks(3)
        service = make_service(_weekly_response())
        await service.weekly_planner(tasks, {"working_hours": "09:00 - 18:00"})
        prompt = service.client.models.generate_content.call_args[1]["contents"]
        for t in tasks:
            assert t["title"] in prompt

    @pytest.mark.asyncio
    async def test_weekly_planner_prompt_contains_availability(self):
        tasks = make_tasks(2)
        availability = {"working_hours": "08:00 - 12:00", "days_off": ["Sunday"]}
        service = make_service(_weekly_response())
        await service.weekly_planner(tasks, availability)
        prompt = service.client.models.generate_content.call_args[1]["contents"]
        assert "08:00 - 12:00" in prompt

    @pytest.mark.asyncio
    async def test_weekly_planner_has_seven_days(self):
        service = make_service(_weekly_response())
        result = await service.weekly_planner(make_tasks(3), {})
        assert len(result.weeklyPlan) == 7

    @pytest.mark.asyncio
    async def test_weekly_planner_empty_tasks(self):
        empty_weekly = json.dumps(
            {
                "weeklyPlan": [
                    {"day": d, "tasks": []}
                    for d in [
                        "Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thursday",
                        "Friday",
                        "Saturday",
                        "Sunday",
                    ]
                ],
                "recommendationSummary": "No hay tareas pendientes.",
            }
        )
        service = make_service(empty_weekly)
        result = await service.weekly_planner([], {})
        assert all(len(day.tasks) == 0 for day in result.weeklyPlan)

    @pytest.mark.asyncio
    async def test_weekly_planner_recommendation_summary_present(self):
        service = make_service(_weekly_response())
        result = await service.weekly_planner(make_tasks(2), {})
        assert len(result.recommendationSummary) > 0

    @pytest.mark.asyncio
    async def test_weekly_planner_invalid_response_raises(self):
        service = make_service("not json")
        with pytest.raises(Exception):
            await service.weekly_planner(make_tasks(1), {})


# ═══════════════════════════════════════════════════════════════════════════════
# 5. task_improve() — Tests unitarios por modo
# ═══════════════════════════════════════════════════════════════════════════════


class TestTaskImproveServiceSubtasks:
    @pytest.mark.asyncio
    async def test_subtasks_mode_returns_subtasks_response(self):
        service = make_service(json.dumps({"subtasks": ["Paso 1", "Paso 2", "Paso 3"]}))
        result = await service.task_improve("Implementar login", "Con JWT", "subtasks")
        assert isinstance(result, SubtasksResponse)

    @pytest.mark.asyncio
    async def test_subtasks_mode_prompt_contains_title(self):
        service = make_service(json.dumps({"subtasks": ["Step 1"]}))
        await service.task_improve("Mi tarea importante", "descripción", "subtasks")
        prompt = service.client.models.generate_content.call_args[1]["contents"]
        assert "Mi tarea importante" in prompt

    @pytest.mark.asyncio
    async def test_subtasks_mode_prompt_contains_description(self):
        service = make_service(json.dumps({"subtasks": ["Step 1"]}))
        await service.task_improve("Tarea", "Descripción muy específica", "subtasks")
        prompt = service.client.models.generate_content.call_args[1]["contents"]
        assert "Descripción muy específica" in prompt

    @pytest.mark.asyncio
    async def test_subtasks_mode_uses_subtasks_schema(self):
        service = make_service(json.dumps({"subtasks": ["Step 1"]}))
        await service.task_improve("Tarea", "", "subtasks")
        call_args = service.client.models.generate_content.call_args[1]
        assert call_args["config"].response_schema == SubtasksResponse

    @pytest.mark.asyncio
    async def test_subtasks_empty_description_uses_placeholder(self):
        service = make_service(json.dumps({"subtasks": ["Step 1"]}))
        await service.task_improve("Tarea sin desc", "", "subtasks")
        prompt = service.client.models.generate_content.call_args[1]["contents"]
        assert "No description provided" in prompt

    @pytest.mark.asyncio
    async def test_subtasks_none_description_uses_placeholder(self):
        service = make_service(json.dumps({"subtasks": ["Step 1"]}))
        # description="" es el default del endpoint; task_improve recibe "" cuando es None
        await service.task_improve("Tarea", "", "subtasks")
        prompt = service.client.models.generate_content.call_args[1]["contents"]
        assert "No description provided" in prompt


class TestTaskImproveServiceEstimate:
    @pytest.mark.asyncio
    async def test_estimate_mode_returns_estimated_time_response(self):
        service = make_service(json.dumps({"estimatedTime": "3h 30m"}))
        result = await service.task_improve("Tarea", "desc", "estimate")
        assert isinstance(result, EstimatedTimeResponse)

    @pytest.mark.asyncio
    async def test_estimate_mode_uses_estimate_schema(self):
        service = make_service(json.dumps({"estimatedTime": "1h"}))
        await service.task_improve("Tarea", "desc", "estimate")
        call_args = service.client.models.generate_content.call_args[1]
        assert call_args["config"].response_schema == EstimatedTimeResponse

    @pytest.mark.asyncio
    async def test_estimate_mode_prompt_contains_title(self):
        service = make_service(json.dumps({"estimatedTime": "2h"}))
        await service.task_improve(
            "Crear API REST completa", "CRUD con auth", "estimate"
        )
        prompt = service.client.models.generate_content.call_args[1]["contents"]
        assert "Crear API REST completa" in prompt


class TestTaskImproveServicePriority:
    @pytest.mark.asyncio
    async def test_priority_mode_returns_suggested_priority_response(self):
        service = make_service(json.dumps({"suggestedPriority": "HIGH"}))
        result = await service.task_improve(
            "Tarea urgente", "Deadline mañana", "priority"
        )
        assert isinstance(result, SuggestedPriorityResponse)

    @pytest.mark.asyncio
    async def test_priority_mode_uses_priority_schema(self):
        service = make_service(json.dumps({"suggestedPriority": "MEDIUM"}))
        await service.task_improve("Tarea", "desc", "priority")
        call_args = service.client.models.generate_content.call_args[1]
        assert call_args["config"].response_schema == SuggestedPriorityResponse

    @pytest.mark.asyncio
    async def test_priority_mode_prompt_contains_title_and_description(self):
        service = make_service(json.dumps({"suggestedPriority": "LOW"}))
        await service.task_improve("Refactorizar código", "Sin deadline", "priority")
        prompt = service.client.models.generate_content.call_args[1]["contents"]
        assert "Refactorizar código" in prompt


class TestTaskImproveServiceAll:
    @pytest.mark.asyncio
    async def test_all_mode_returns_improve_all_response(self):
        service = make_service(
            json.dumps(
                {
                    "subtasks": ["Diseñar", "Implementar"],
                    "estimatedTime": "4h",
                    "suggestedPriority": "HIGH",
                }
            )
        )
        result = await service.task_improve("Feature completa", "Módulo auth", "all")
        assert isinstance(result, ImproveAllResponse)

    @pytest.mark.asyncio
    async def test_all_mode_uses_improve_all_schema(self):
        service = make_service(
            json.dumps(
                {
                    "subtasks": ["ok"],
                    "estimatedTime": "1h",
                    "suggestedPriority": "LOW",
                }
            )
        )
        await service.task_improve("Tarea", "desc", "all")
        call_args = service.client.models.generate_content.call_args[1]
        assert call_args["config"].response_schema == ImproveAllResponse

    @pytest.mark.asyncio
    async def test_unknown_mode_falls_back_to_all(self):
        """Cualquier modo no reconocido cae en el else (all)."""
        service = make_service(
            json.dumps(
                {
                    "subtasks": ["ok"],
                    "estimatedTime": "1h",
                    "suggestedPriority": "LOW",
                }
            )
        )
        result = await service.task_improve("Tarea", "desc", "nonexistent_mode")
        assert isinstance(result, ImproveAllResponse)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Error Handling — cuando Gemini falla o retorna datos inválidos
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlannerServiceErrorHandling:
    """Tests de resiliencia cuando Gemini retorna respuestas inesperadas."""

    @pytest.mark.asyncio
    async def test_organize_gemini_returns_empty_string_raises(self):
        service = make_service("")
        with pytest.raises(Exception):
            await service.organize_tasks(make_tasks(1))

    @pytest.mark.asyncio
    async def test_organize_gemini_returns_wrong_schema_raises(self):
        """JSON válido pero con schema incorrecto debe lanzar ValidationError."""
        service = make_service(json.dumps({"wrong_field": "wrong_value"}))
        with pytest.raises((ValidationError, Exception)):
            await service.organize_tasks(make_tasks(1))

    @pytest.mark.asyncio
    async def test_calendar_gemini_invalid_json_raises(self):
        service = make_service("{invalid")
        with pytest.raises(Exception):
            await service.ai_calendar_planner(make_tasks(1), [])

    @pytest.mark.asyncio
    async def test_weekly_gemini_invalid_json_raises(self):
        service = make_service("null")
        with pytest.raises(Exception):
            await service.weekly_planner(make_tasks(1), {})

    @pytest.mark.asyncio
    async def test_improve_subtasks_invalid_json_raises(self):
        service = make_service("not_json")
        with pytest.raises(Exception):
            await service.task_improve("Tarea", "desc", "subtasks")

    @pytest.mark.asyncio
    async def test_improve_estimate_wrong_schema_raises(self):
        service = make_service(json.dumps({"otherField": "value"}))
        with pytest.raises((ValidationError, Exception)):
            await service.task_improve("Tarea", "desc", "estimate")

    @pytest.mark.asyncio
    async def test_gemini_client_exception_propagates(self):
        """Si genai.Client.models.generate_content lanza excepción, debe propagarse."""
        service = make_service("")
        service.client.models.generate_content.side_effect = RuntimeError(
            "Network error"
        )
        with pytest.raises(RuntimeError, match="Network error"):
            await service.organize_tasks(make_tasks(1))
