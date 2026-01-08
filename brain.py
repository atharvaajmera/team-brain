import os
from dotenv import load_dotenv
from ingest import get_threads_from_channel
from memory import add_messages

load_dotenv()

def builder():
    CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID")
    if not CHANNEL_ID:
        raise ValueError("SLACK_CHANNEL_ID environment variable is not set")
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
        thread_id = msg.get('thread_ts', ts)
        full_text = f"User: {user}\nMessage: {text}\nTimestamp: {ts}"
        print(full_text)
        text_to_embed=f"{text}"
        texts.append(text_to_embed)
        ids.append(f"{user}_{ts}")
        metadatas.append({
            "user": user,
            "ts": ts,
            "text": text,
            "thread_id": thread_id
        })
    add_messages(texts, ids, metadatas)

if __name__ == "__main__":
    builder()
    print("Threads saved to chromadb successfully.")