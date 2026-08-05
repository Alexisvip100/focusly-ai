from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.services.ai.planner import (
    AIPlannerService,
    TasksOrganizeResponse,
    CalendarPlannerResponse,
    WeeklyPlanResponse,
)

router = APIRouter(prefix="/planner", tags=["planner"])


def get_planner_service() -> AIPlannerService:
    return AIPlannerService()


# ─── Request Schemas ───────────────────────────────────────────────────────────


class OrganizeTasksRequest(BaseModel):
    tasks: list[dict[str, object]]


class CalendarPlannerRequest(BaseModel):
    tasks: list[dict[str, object]]
    free_slots: list[dict[str, object]]


class WeeklyPlannerRequest(BaseModel):
    tasks: list[dict[str, object]]
    availability: dict[str, object] | None = None


class TaskImproveRequest(BaseModel):
    title: str
    description: str | None = ""
    mode: str  # subtasks, estimate, priority, all


# ─── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/organize", response_model=TasksOrganizeResponse)
async def organize_tasks(
    body: OrganizeTasksRequest,
    planner: AIPlannerService = Depends(get_planner_service),
):
    try:
        plan = await planner.organize_tasks(body.tasks)
        return plan
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to organize tasks: {str(e)}"
        )


@router.post("/calendar", response_model=CalendarPlannerResponse)
async def calendar_planner(
    body: CalendarPlannerRequest,
    planner: AIPlannerService = Depends(get_planner_service),
):
    try:
        plan = await planner.ai_calendar_planner(body.tasks, body.free_slots)
        return plan
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to plan calendar events: {str(e)}"
        )


@router.post("/weekly", response_model=WeeklyPlanResponse)
async def weekly_planner(
    body: WeeklyPlannerRequest,
    planner: AIPlannerService = Depends(get_planner_service),
):
    try:
        availability = body.availability or {"working_hours": "09:00 - 18:00"}
        plan = await planner.weekly_planner(body.tasks, availability)
        return plan
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate weekly plan: {str(e)}"
        )


@router.post("/improve")
async def task_improve(
    body: TaskImproveRequest,
    planner: AIPlannerService = Depends(get_planner_service),
):
    try:
        result = await planner.task_improve(
            body.title, body.description or "", body.mode
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to suggest task improvements: {str(e)}"
        )
