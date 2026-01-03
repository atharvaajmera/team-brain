import chromadb
from chromadb.config import Settings
import json

client = chromadb.PersistentClient(path="./chroma_db",settings=Settings(anonymized_telemetry=False))

collection = client.get_or_create_collection(name="slack_archive")

def add_messages(texts, ids, metadatas):
    collection.add(documents=texts, ids=ids, metadatas=metadatas)
    print("Got the threads from ingest and added to memory.")
