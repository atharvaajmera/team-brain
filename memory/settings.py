import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load env vars once here
load_dotenv()


@dataclass
class Settings:
    SLACK_BOT_TOKEN: str = os.getenv("SLACK_BOT_TOKEN", "")
    SLACK_SYNC_TOKEN: str = (
        os.getenv("SLACK_SYNC_TOKEN", "")
        or os.getenv("SLACK_USER_TOKEN", "")
        or os.getenv("SLACK_BOT_TOKEN", "")
    )
    SLACK_CHANNEL_ID_AUTO: str = os.getenv("SLACK_CHANNEL_ID_AUTO", "")
    SLACK_CHANNEL_ID: str = os.getenv("SLACK_CHANNEL_ID", "")
    SLACK_WORKSPACE: str = os.getenv("SLACK_WORKSPACE", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    MODEL: str = os.getenv("MODEL", "llama3.2")


settings = Settings()
