import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GOOGLE_GENERATIVE_AI_API_KEY: str = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
        or ""
    )

settings = Settings()
