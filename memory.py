import chromadb
from chromadb.config import Settings
import json

client = chromadb.PersistentClient(path="./chroma_db",settings=Settings(anonymized_telemetry=False))

collection = client.get_or_create_collection(name="slack_archive")

def add_messages(texts, ids, metadatas):
    collection.upsert(documents=texts, ids=ids, metadatas=metadatas)
    print("Got the threads from ingest and added to memory.")

def query_text(query):
    results = collection.query(
        query_texts=[query],
        n_results=5
    )

    if not results['documents']:
        return []
    
    ids = results['ids'][0]
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    outputs = []
   
    for doc,metadata,id,distance in zip(documents,metadatas,ids,distances):
        output = {
            "id": id,
            "document": doc,
            "metadata": metadata,
            "distance": distance
        }
        outputs.append(output)
    return outputs
