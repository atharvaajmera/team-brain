from ingest import get_threads_from_channel
from memory import add_messages

def builder():
    CHANNEL_ID="C0A4LV0HZJ7"
    messages = get_threads_from_channel(CHANNEL_ID, limit=50)
    texts=[]
    ids=[]
    metadatas=[]
    
    for msg in messages:
        text = msg.get('text', '')
        if not text.strip():
            continue
        user = msg.get('user', 'unknown_user')
        ts = msg.get('ts', 'no_ts')
        thread_ts = msg.get('thread_ts', ts)
        full_text = f"User: {user}\nMessage: {text}\nTimestamp: {ts}"
        texts.append(full_text)
        ids.append(f"{user}_{ts}")
        metadatas.append({
            "user": user,
            "ts": ts,
            "text": text,
            "thread_ts": thread_ts
        })
    add_messages(texts, ids, metadatas)

if __name__ == "__main__":
    builder()
    print("Threads saved to chromadb successfully.")