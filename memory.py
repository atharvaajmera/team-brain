import chromadb
from chromadb.config import Settings
import json

client = chromadb.PersistentClient(path="./chroma_db",settings=Settings(anonymized_telemetry=False))

collection = client.get_or_create_collection(name="slack_archive")

def add_messages(texts, ids, metadatas):
    collection.upsert(documents=texts, ids=ids, metadatas=metadatas)
    print("Got the threads from ingest and added to memory.")


def query_text_phase_1(query):
    results = collection.query(
        query_texts=[query],
        n_results=1
    )

    if not results['documents']:
        return []
    
    ids = results['ids'][0]
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    outputs = []
   
    for doc,metadata,id,distance in zip(documents,metadatas,ids,distances):
        if distance < 1.2:
            output = {
                "id": id,
                "document": doc,
                "metadata": metadata,
                "distance": distance
            }
            outputs.append(output)
    return outputs

def get_top_convo_id(query):
    anchor=query_text_phase_1(query)
    top_conversation_id=anchor[0]['metadata']['thread_id'] if anchor else None
    return top_conversation_id

def query_text_phase_2(query):
    top_convo_id=get_top_convo_id(query)
    if not top_convo_id:
        return []
    results = collection.get(
        where={
            "thread_id": top_convo_id
        }
    )
    if not results['documents']:
        return []
    ids = results['ids']
    documents = results['documents']
    metadatas = results['metadatas']
    outputs = []
    for doc,metadata,id in zip(documents,metadatas,ids):
        output = {
            "id": id,
            "document": doc,
            "metadata": metadata
        }
        outputs.append(output)
    outputs.sort(key=lambda x: float(x['metadata']['ts']))
    return outputs