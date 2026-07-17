# focusly-ai

Stateless FastAPI microservice hosting Gemini/Claude-backed AI capabilities and productivity planners for Focusly.

## Requisitos

- Docker y Docker Compose (opción recomendada), **o**
- Python 3.11+ si prefieres correrlo sin Docker

## Configuración

1. Copia el archivo de ejemplo y completa tus API keys:

   ```bash
   cp .env.example .env
   ```

2. Edita `.env` con tus valores reales:
   - `GOOGLE_GENERATIVE_AI_API_KEY` (o `GEMINI_API_KEY`)
   - `ANTHROPIC_API_KEY`
   - `PORT` (por defecto `8001`)

## Levantar el servicio (Docker — forma recomendada)

```bash
docker-compose up --build
```

Esto construye la imagen y levanta el servicio en `http://localhost:8001` (o el puerto que definas en `PORT`). Para correrlo en segundo plano:

```bash
docker-compose up --build -d
```

Para detenerlo:

```bash
docker-compose down
```

## Levantar el servicio (local, sin Docker)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

`--reload` recarga el servidor automáticamente en cada cambio de código (solo recomendado para desarrollo local).

## Verificar que está corriendo

```bash
curl http://localhost:8001/
```

Debe responder:

```json
{ "status": "healthy", "service": "focusly-ai", "version": "1.0.0" }
```

La documentación interactiva de la API (Swagger) está disponible en `http://localhost:8001/docs`.

## Endpoints principales

- `POST /ai/chat` — chat conversacional
- `POST /ai/analyze-patterns` — análisis de patrones de productividad
- `POST /ai/planner/organize` — organización de tareas
- `POST /ai/planner/calendar` — planificación de calendario
- `POST /ai/planner/weekly` — plan semanal
- `POST /ai/planner/improve` — mejora de planes existentes
