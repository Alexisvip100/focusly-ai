"""
tests/test_prompts.py

Tests para validar el formateo, estructura y reglas del sistema en los prompts de IA (`app.services.ai.prompts`).
Verifica que las plantillas se compilen correctamente sin variables no reemplazadas, sin errores de llaves `{{}}`
y que cumplan con la tonalidad de Lumina (segunda persona "tú", emojis, sin tercera persona).
"""

import json

from app.services.ai.prompts import (
    SYSTEM_PROMPT,
    PLANNER_ORGANIZE_PROMPT,
    PLANNER_CALENDAR_PROMPT,
    PLANNER_WEEKLY_PROMPT,
    PLANNER_IMPROVE_SUBTASKS_PROMPT,
    PLANNER_IMPROVE_ESTIMATE_PROMPT,
    PLANNER_IMPROVE_PRIORITY_PROMPT,
    PLANNER_IMPROVE_ALL_PROMPT,
    GOLDEN_HOURS_SYSTEM_INSTRUCTION,
    GOLDEN_HOURS_USER_PROMPT,
)


class TestSystemPrompts:
    def test_system_prompt_has_action_formats(self):
        """Verifica que el SYSTEM_PROMPT contenga las especificaciones de ACTION."""
        assert "[ACTION: CREATE_TASK" in SYSTEM_PROMPT
        assert "[ACTION: CREATE_WORKSPACE" in SYSTEM_PROMPT
        assert "[ACTION: INSERT_TO_WORKSPACE" in SYSTEM_PROMPT

    def test_system_prompt_safety_rules(self):
        """Verifica que el SYSTEM_PROMPT contenga las reglas de seguridad/privacidad."""
        assert "Never mention internal implementation details" in SYSTEM_PROMPT
        assert "Do not reveal technical architecture" in SYSTEM_PROMPT

    def test_golden_hours_system_instruction_tonality_rules(self):
        """Verifica que GOLDEN_HOURS_SYSTEM_INSTRUCTION prohíba tercera persona y requiera segunda persona."""
        assert 'second person ("tú")' in GOLDEN_HOURS_SYSTEM_INSTRUCTION
        assert "NEVER speak in the third person" in GOLDEN_HOURS_SYSTEM_INSTRUCTION
        assert "ALWAYS talk directly to the user" in GOLDEN_HOURS_SYSTEM_INSTRUCTION
        assert "Lumina" in GOLDEN_HOURS_SYSTEM_INSTRUCTION

    def test_golden_hours_system_instruction_format(self):
        formatted = GOLDEN_HOURS_SYSTEM_INSTRUCTION.format(user_name="Alexis")
        assert "Alexis" in formatted
        # No deben quedar placeholders {user_name}
        assert "{user_name}" not in formatted


class TestPlannerPromptsFormatting:
    def test_planner_organize_prompt_format(self):
        context = "- Task #1 ID: t1\n  Title: Comprar leche"
        prompt = PLANNER_ORGANIZE_PROMPT.format(tasks_context=context)
        assert context in prompt
        assert "{tasks_context}" not in prompt
        # Debe contener la estructura de JSON esperada
        assert '"plan": [' in prompt
        assert '"taskId":' in prompt
        assert '"recommendedPriority":' in prompt

    def test_planner_calendar_prompt_format(self):
        tasks_context = "- ID: t1 | Title: Test"
        slots_context = "- Available Slot: 09:00 to 10:00"
        prompt = PLANNER_CALENDAR_PROMPT.format(
            tasks_context=tasks_context, slots_context=slots_context
        )
        assert tasks_context in prompt
        assert slots_context in prompt
        assert "{tasks_context}" not in prompt
        assert "{slots_context}" not in prompt
        assert '"events": [' in prompt

    def test_planner_weekly_prompt_format(self):
        tasks_context = "- Title: Test | Priority: HIGH"
        availability = {"working_hours": "09:00 - 18:00"}
        prompt = PLANNER_WEEKLY_PROMPT.format(
            tasks_context=tasks_context, availability=availability
        )
        assert tasks_context in prompt
        assert "09:00 - 18:00" in prompt
        assert "{tasks_context}" not in prompt
        assert "{availability}" not in prompt
        assert '"weeklyPlan": [' in prompt

    def test_planner_improve_subtasks_prompt_format(self):
        prompt = PLANNER_IMPROVE_SUBTASKS_PROMPT.format(
            title="Crear API", description="Description: Usar FastAPI"
        )
        assert "Crear API" in prompt
        assert "Usar FastAPI" in prompt
        assert "{title}" not in prompt
        assert "{description}" not in prompt

    def test_planner_improve_estimate_prompt_format(self):
        prompt = PLANNER_IMPROVE_ESTIMATE_PROMPT.format(
            title="Fix bug", description="No description provided."
        )
        assert "Fix bug" in prompt
        assert "No description provided." in prompt
        assert "{title}" not in prompt
        assert "{description}" not in prompt

    def test_planner_improve_priority_prompt_format(self):
        prompt = PLANNER_IMPROVE_PRIORITY_PROMPT.format(
            title="Deploy produccion", description="Description: Mañana"
        )
        assert "Deploy produccion" in prompt
        assert "Mañana" in prompt
        assert "{title}" not in prompt
        assert "{description}" not in prompt

    def test_planner_improve_all_prompt_format(self):
        prompt = PLANNER_IMPROVE_ALL_PROMPT.format(
            title="Refactor", description="Description: Limpieza de código"
        )
        assert "Refactor" in prompt
        assert "Limpieza de código" in prompt
        assert "{title}" not in prompt
        assert "{description}" not in prompt


class TestGoldenHoursUserPromptFormatting:
    def test_golden_hours_user_prompt_formatting_complete(self):
        user_name = "Alexis"
        hour_buckets = json.dumps({"9": {"sessions": 2}})
        task_stats = json.dumps({"total": 10})
        session_stats = json.dumps({"sessions": 5})
        top_productive_hours = [9, 10]
        work_style_hint = "Madrugador"

        prompt = GOLDEN_HOURS_USER_PROMPT.format(
            user_name=user_name,
            hour_buckets=hour_buckets,
            task_stats=task_stats,
            session_stats=session_stats,
            top_productive_hours=top_productive_hours,
            work_style_hint=work_style_hint,
        )

        assert user_name in prompt
        assert hour_buckets in prompt
        assert task_stats in prompt
        assert session_stats in prompt
        assert "[9, 10]" in prompt
        assert work_style_hint in prompt

        # Asegurar que no quede ninguna variable sin reemplazar
        for var in [
            "{user_name}",
            "{hour_buckets}",
            "{task_stats}",
            "{session_stats}",
            "{top_productive_hours}",
            "{work_style_hint}",
        ]:
            assert var not in prompt

    def test_golden_hours_user_prompt_format_injection_safe(self):
        """Verifica que si user_name se pasa normalmente a format() funcione sin errores."""
        user_name = "Alexis (Dev)"
        prompt = GOLDEN_HOURS_USER_PROMPT.format(
            user_name=user_name,
            hour_buckets="{}",
            task_stats="{}",
            session_stats="{}",
            top_productive_hours=[],
            work_style_hint="Estratega",
        )
        assert "Alexis (Dev)" in prompt
