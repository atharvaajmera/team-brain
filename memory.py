import chromadb
from chromadb.config import Settings
import time
from datetime import datetime, timedelta

client = chromadb.PersistentClient(path="./chroma_db",settings=Settings(anonymized_telemetry=False))

collection = client.get_or_create_collection(name="slack_archive")

def add_messages(texts, ids, metadatas):
    collection.upsert(documents=texts, ids=ids, metadatas=metadatas)
    print("Got the threads from ingest and added to memory.")

timeline=["recent","today","yesterday","this week","last week","this month","last month","this year","last year","all time"]
timeline.sort(key=lambda x: -len(x)) 

aggregation=["issues","bugs","tasks","features","improvements","questions","discussions","announcements","updates"]
aggregation.sort(key=lambda x: -len(x))

def get_unix_timestamp(timeline_key):
    now = datetime.now()
    if timeline_key == "recent":
        target_date = now - timedelta(hours=24)
    elif timeline_key == "today":
        target_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif timeline_key == "yesterday":
        target_date = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif timeline_key == "this week":
        target_date = now - timedelta(days=now.weekday())
        target_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    elif timeline_key == "last week":
        target_date = now - timedelta(days=now.weekday() + 7)
        target_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    elif timeline_key == "this month":
        target_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif timeline_key == "last month":
        first_day_this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        target_date = (first_day_this_month - timedelta(days=1)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif timeline_key == "this year":
        target_date = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    elif timeline_key == "last year":
        target_date = now.replace(year=now.year - 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    elif timeline_key == "all time":
        return 0
    else:
        return None
    return int(target_date.timestamp())

def analyze_query_intent(query):
    user_query=query.lower()
    intent = {"timeline": None, "aggregation": None, "filter_timeline": None}
    for time in timeline:
        if time in user_query:
            intent["timeline"] = time
            intent["filter_timeline"] = get_unix_timestamp(time)
            break 

    for agg in aggregation:
        if agg in user_query:
            intent["aggregation"] = agg
            break

    return intent

def build_chroma_filter(query):
    intent=analyze_query_intent(query)
    chroma_filter={}

    if intent['filter_timeline']:
        chroma_filter['ts']={"$gte":intent['filter_timeline']}

    if intent['aggregation']:
        print("Aggregation detected:", intent['aggregation'])
        # Implement aggregation logic as needed

    if not chroma_filter:
        chroma_filter=None

    return chroma_filter

def retrieve_candidates(query, with_filter=True):
    chroma_filter = build_chroma_filter(query) if with_filter else None
    n_results = 10 if chroma_filter else 5  
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=chroma_filter
    )

    if not results['documents'] or not results['documents'][0]:
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

def select_anchor(candidates, mode):
    if not candidates:
        return None

    sorted_candidates = sorted(candidates, key=lambda x: x['distance'])
    best = sorted_candidates[0]

    if mode == "NORMAL":
        if best['distance'] < 1.2:
            return best
        return None
    elif mode == "FALLBACK":
        return best
    else:
        return None

def get_top_convo_id(query):
    candidates = retrieve_candidates(query, with_filter=True)

    anchor = select_anchor(candidates, mode="NORMAL")
    
    if anchor:
        return anchor['metadata']['thread_id']

    if build_chroma_filter(query):
        print("No results with temporal filter. Retrying without filter")

        candidates = retrieve_candidates(query, with_filter=False)
        anchor = select_anchor(candidates, mode="FALLBACK")
        if anchor:
            return anchor['metadata']['thread_id']

    return None

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