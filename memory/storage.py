import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "None"

import chromadb
from chromadb.config import Settings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = REPO_ROOT / "chroma_db"

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH),
    settings=Settings(anonymized_telemetry=False),
)

collection = client.get_or_create_collection(name="slack_archive")

def add_messages(texts, ids, metadatas):
    collection.upsert(documents=texts, ids=ids, metadatas=metadatas)
    print("Got the threads from ingest and added to memory.")

def reset_collection():
    global collection
    try:
        client.delete_collection(name="slack_archive")
    except Exception:
        pass
    collection = client.get_or_create_collection(name="slack_archive")
