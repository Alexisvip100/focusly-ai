from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.ai.ai import router as ai_router
from app.routes.ai.planner import router as planner_router

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

# Include Routers
app.include_router(ai_router, prefix="/ai")
app.include_router(planner_router, prefix="/ai")


@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "focusly-ai", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}