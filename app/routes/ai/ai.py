from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import httpx
import json

from app.config import settings
from app.services.ai.prompts import GOLDEN_HOURS_SYSTEM_INSTRUCTION, GOLDEN_HOURS_USER_PROMPT

router = APIRouter(tags=["ai"])

# ─── Schemas ──────────────────────────────────────────────────────────────────

class MessageSchema(BaseModel):
    role: str
    content: str

class ChatRequestSchema(BaseModel):
    messages: List[MessageSchema]
    system_context: str
    model: Optional[str] = "gemini-2.5-flash-lite"

class AnalyzePatternsRequest(BaseModel):
    user_name: str
    hour_buckets: Dict[str, Any]
    task_stats: Dict[str, Any]
    session_stats: Dict[str, Any]
    top_productive_hours: List[int]
    work_style_hint: str

# ─── Gemini Stream Parser ─────────────────────────────────────────────────────

class GeminiStreamParser:
    def __init__(self):
        self.buffer = ""

    def feed(self, chunk: str):
        self.buffer += chunk
        self.buffer = self.buffer.lstrip("[\r\n, ")
        
        while self.buffer:
            if not self.buffer.startswith("{"):
                self.buffer = self.buffer.lstrip("\r\n, ]")
                if not self.buffer:
                    break
                if not self.buffer.startswith("{"):
                    self.buffer = ""
                    break
            
            brace_count = 0
            in_string = False
            escape = False
            end_idx = -1
            
            for idx, char in enumerate(self.buffer):
                if char == '"' and not escape:
                    in_string = not in_string
                elif char == '\\' and in_string:
                    escape = not escape
                else:
                    escape = False
                
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = idx
                            break
            
            if end_idx != -1:
                obj_str = self.buffer[:end_idx + 1]
                self.buffer = self.buffer[end_idx + 1:].lstrip("\r\n, ")
                try:
                    obj = json.loads(obj_str)
                    candidates = obj.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        if parts:
                            text = parts[0].get("text", "")
                            if text:
                                yield text
                except Exception:
                    pass
            else:
                break

# ─── Streaming Assistant ──────────────────────────────────────────────────────

async def stream_gemini(payload: dict, model: str):
    api_key = settings.GOOGLE_GENERATIVE_AI_API_KEY
    if not api_key:
        yield "Error: GEMINI_API_KEY is not configured in focusly-ai settings."
        return

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?key={api_key}"
    parser = GeminiStreamParser()
    
    async with httpx.AsyncClient() as client:
        try:
            async with client.stream("POST", url, json=payload, timeout=60.0) as r:
                if r.status_code != 200:
                    error_text = await r.aread()
                    yield f"Error calling Gemini API: {r.status_code} - {error_text.decode('utf-8', errors='ignore')}"
                    return
                
                async for chunk in r.aiter_text():
                    for text in parser.feed(chunk):
                        yield text
        except Exception as e:
            yield f"\nStreaming error: {str(e)}"

# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat_endpoint(body: ChatRequestSchema):
    if not body.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty")
        
    latest_user_message = body.messages[-1].content
    
    # Restructure content history into Gemini format
    gemini_contents = []
    for m in body.messages[:-1]:
        gemini_contents.append({
            "role": "user" if m.role == "user" else "model",
            "parts": [{"text": m.content}]
        })
    
    # Append latest user message
    gemini_contents.append({
        "role": "user",
        "parts": [{"text": latest_user_message}]
    })

    payload = {
        "contents": gemini_contents,
        "systemInstruction": {
            "parts": [{"text": body.system_context}]
        }
    }
    
    return StreamingResponse(
        stream_gemini(payload, body.model),
        media_type="text/plain"
    )

@router.post("/analyze-patterns")
async def analyze_patterns_endpoint(body: AnalyzePatternsRequest):
    api_key = settings.GOOGLE_GENERATIVE_AI_API_KEY
    if not api_key:
        raise HTTPException(status_code=500, detail="Gemini API key is not configured in focusly-ai")

    system_instruction = GOLDEN_HOURS_SYSTEM_INSTRUCTION.format(user_name=body.user_name)

    user_prompt = GOLDEN_HOURS_USER_PROMPT.format(
        user_name=body.user_name,
        hour_buckets=json.dumps(body.hour_buckets, indent=2),
        task_stats=json.dumps(body.task_stats, indent=2),
        session_stats=json.dumps(body.session_stats, indent=2),
        top_productive_hours=body.top_productive_hours,
        work_style_hint=body.work_style_hint
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": user_prompt}]
        }],
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(url, json=payload, timeout=30.0)
            if r.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Gemini API returned code {r.status_code}")
            res_json = r.json()
            text_response = res_json["candidates"][0]["content"]["parts"][0]["text"]
            
            analysis = json.loads(text_response.strip())
            return {
                "success": True,
                "data": analysis
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error analyzing patterns in focusly-ai: {str(e)}")
