from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.ai.ai import router as ai_router
from app.routes.ai.planner import router as planner_router
from app.security import require_internal_token

app = FastAPI(
    title="Focusly AI Microservice",
    description="Stateless microservice hosting Gemini capabilities and productivity planners.",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers (health checks stay public; the AI endpoints don't)
app.include_router(
    ai_router, prefix="/ai", dependencies=[Depends(require_internal_token)]
)
app.include_router(
    planner_router, prefix="/ai", dependencies=[Depends(require_internal_token)]
)


@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "focusly-ai", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}
