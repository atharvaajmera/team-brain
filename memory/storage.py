import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "None"

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings

logger = logging.getLogger("storage")

REPO_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = REPO_ROOT / "chroma_db"
CHROMA_HOST = os.getenv("CHROMA_HOST", "").strip()
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

if CHROMA_HOST:
    logger.info("Using remote ChromaDB at %s:%s", CHROMA_HOST, CHROMA_PORT)
    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )
else:
    logger.info("Using local ChromaDB at %s", CHROMA_PATH)
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )

collection = client.get_or_create_collection(name="slack_archive")


def add_messages(texts, ids, metadatas):
    collection.upsert(documents=texts, ids=ids, metadatas=metadatas)
    logger.info("Got the threads from ingest and added to memory.")


def reset_collection():
    global collection
    try:
        client.delete_collection(name="slack_archive")
    except Exception:
        pass
    collection = client.get_or_create_collection(name="slack_archive")
