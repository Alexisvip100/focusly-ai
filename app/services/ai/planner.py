import asyncio
import os
from typing import Any
from pydantic import BaseModel
from google import genai
from google.genai import types

from app.config import settings
from app.services.ai.models import light_model, sdk_thinking_config, strong_model
from app.services.ai.prompts import (
    PLANNER_ORGANIZE_PROMPT,
    PLANNER_CALENDAR_PROMPT,
    PLANNER_WEEKLY_PROMPT,
    PLANNER_IMPROVE_SUBTASKS_PROMPT,
    PLANNER_IMPROVE_ESTIMATE_PROMPT,
    PLANNER_IMPROVE_PRIORITY_PROMPT,
    PLANNER_IMPROVE_ALL_PROMPT,
)

# ─── Pydantic schemas for Gemini Structured Output ──────────────────────────────


class TaskPlanItem(BaseModel):
    taskId: str
    recommendedPriority: str  # HIGH, MEDIUM, LOW
    suggestedOrder: int
    reason: str
    suggestedDate: str | None = None
    estimatedTime: str | None = None


class TasksOrganizeResponse(BaseModel):
    plan: list[TaskPlanItem]


class TimeBlockItem(BaseModel):
    taskId: str | None = None
    title: str
    startTime: str  # ISO-8601 string
    endTime: str  # ISO-8601 string
    reason: str


class CalendarPlannerResponse(BaseModel):
    events: list[TimeBlockItem]


class WeeklyPlanDayItem(BaseModel):
    day: str  # Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday
    tasks: list[str]


class WeeklyPlanResponse(BaseModel):
    weeklyPlan: list[WeeklyPlanDayItem]
    recommendationSummary: str


class SubtasksResponse(BaseModel):
    subtasks: list[str]


class EstimatedTimeResponse(BaseModel):
    estimatedTime: str


class SuggestedPriorityResponse(BaseModel):
    suggestedPriority: str  # HIGH, MEDIUM, LOW


class ImproveAllResponse(BaseModel):
    subtasks: list[str]
    estimatedTime: str
    suggestedPriority: str


# ─── Service Class ─────────────────────────────────────────────────────────────


class AIPlannerService:
    def __init__(self):
        api_key = (
            settings.GOOGLE_GENERATIVE_AI_API_KEY
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
        )
        self.client = genai.Client(api_key=api_key)
        # Planning (order, time blocks, the week) needs some reasoning.
        self.model = strong_model()
        # Subtasks, an estimate or a priority for one task don't.
        self.light_model = light_model()

    async def _generate(
        self, prompt: str, schema: type[BaseModel], *, light: bool = False
    ):
        """One JSON answer from Gemini, without blocking the event loop and
        with the thinking (billed as output) capped."""
        model = self.light_model if light else self.model
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(
                **sdk_thinking_config(model, structured=True)
            ),
        )
        return await asyncio.wait_for(
            self.client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            ),
            timeout=settings.REQUEST_TIMEOUT_SECONDS,
        )

    async def organize_tasks(
        self, tasks: list[dict[str, Any]]
    ) -> TasksOrganizeResponse:
        tasks_context = ""
        for i, t in enumerate(tasks):
            tasks_context += (
                f"- Task #{i + 1} ID: {t.get('id')}\n"
                f"  Title: {t.get('title')}\n"
                f"  Description: {t.get('description') or 'No description'}\n"
                f"  Current Priority: {t.get('priority') or 'N/A'}\n"
                f"  Deadline: {t.get('deadline') or 'None'}\n"
                f"  Status: {t.get('status') or 'Todo'}\n"
                f"  Estimated Effort/Time: {t.get('estimatedTime') or 'N/A'}\n\n"
            )

        prompt = PLANNER_ORGANIZE_PROMPT.format(tasks_context=tasks_context)
        response = await self._generate(prompt, TasksOrganizeResponse)
        return TasksOrganizeResponse.model_validate_json(response.text)

    async def ai_calendar_planner(
        self, tasks: list[dict[str, Any]], free_slots: list[dict[str, Any]]
    ) -> CalendarPlannerResponse:
        tasks_context = "\n".join(
            [
                f"- ID: {t.get('id')} | Title: {t.get('title')} | Duration: {t.get('duration') or '30 mins'} | Priority: {t.get('priority')}"
                + (f" | Deadline: {t['deadline']}" if t.get("deadline") else "")
                for t in tasks
            ]
        )

        slots_context = "\n".join(
            [
                f"- Available Slot: {s.get('start')} to {s.get('end')}"
                for s in free_slots
            ]
        )

        prompt = PLANNER_CALENDAR_PROMPT.format(
            tasks_context=tasks_context, slots_context=slots_context
        )
        response = await self._generate(prompt, CalendarPlannerResponse)
        return CalendarPlannerResponse.model_validate_json(response.text)

    async def weekly_planner(
        self, tasks: list[dict[str, Any]], availability: dict[str, Any]
    ) -> WeeklyPlanResponse:
        tasks_context = "\n".join(
            [
                f"- Title: {t.get('title')} | Priority: {t.get('priority')} | Deadline: {t.get('deadline')}"
                for t in tasks
            ]
        )

        prompt = PLANNER_WEEKLY_PROMPT.format(
            tasks_context=tasks_context, availability=availability
        )
        response = await self._generate(prompt, WeeklyPlanResponse)
        return WeeklyPlanResponse.model_validate_json(response.text)

    async def task_improve(self, title: str, description: str, mode: str) -> Any:
        desc_info = (
            f"Description: {description}" if description else "No description provided."
        )

        if mode == "subtasks":
            prompt = PLANNER_IMPROVE_SUBTASKS_PROMPT.format(
                title=title, description=desc_info
            )
            response = await self._generate(prompt, SubtasksResponse, light=True)
            return SubtasksResponse.model_validate_json(response.text)

        elif mode == "estimate":
            prompt = PLANNER_IMPROVE_ESTIMATE_PROMPT.format(
                title=title, description=desc_info
            )
            response = await self._generate(prompt, EstimatedTimeResponse, light=True)
            return EstimatedTimeResponse.model_validate_json(response.text)

        elif mode == "priority":
            prompt = PLANNER_IMPROVE_PRIORITY_PROMPT.format(
                title=title, description=desc_info
            )
            response = await self._generate(
                prompt, SuggestedPriorityResponse, light=True
            )
            return SuggestedPriorityResponse.model_validate_json(response.text)

        else:  # mode == "all"
            prompt = PLANNER_IMPROVE_ALL_PROMPT.format(
                title=title, description=desc_info
            )
            response = await self._generate(prompt, ImproveAllResponse, light=True)
            return ImproveAllResponse.model_validate_json(response.text)
