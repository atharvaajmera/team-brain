import os
import json
from pathlib import Path
from dotenv import load_dotenv
from ingest import get_threads_from_channel
from memory.storage import add_messages

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
SYNC_STATE_FILE = REPO_ROOT / "config" / "sync_state.json"


def _load_last_ts():
    if SYNC_STATE_FILE.exists():
        try:
            with open(SYNC_STATE_FILE) as f:
                return json.load(f).get("last_ts")
        except Exception:
            pass
    return None


def _save_last_ts(ts):
    SYNC_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SYNC_STATE_FILE, "w") as f:
        json.dump({"last_ts": ts}, f)


def builder(full_reindex=False):
    CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID_AUTO")
    last_ts = None if full_reindex else _load_last_ts()

    if last_ts:
        print(f"Incremental sync: fetching messages after ts={last_ts}")
    else:
        print("Full sync: fetching all messages")

    messages = get_threads_from_channel(CHANNEL_ID, after_ts=last_ts)
    if not messages:
        print("No new messages found.")
        return

    texts, ids, metadatas = [], [], []
    max_ts = 0

    for msg in messages:
        text = msg.get("text", "")
        if not text.strip():
            continue
        author = msg.get("author", msg.get("user", "unknown_user"))
        ts = float(msg.get("ts", 0))
        thread_id = float(msg.get("thread_ts", ts))
        max_ts = max(max_ts, ts)

        # Enriched embedding: includes author for person-targeted queries
        text_to_embed = f"{author}: {text}"

        texts.append(text_to_embed)
        ids.append(f"{author}_{ts}")
        metadatas.append({
            "author": author, "ts": ts,
            "text": text, "thread_id": thread_id,
        })

    if texts:
        add_messages(texts, ids, metadatas)
        _save_last_ts(max_ts)
        print(f"Ingested {len(texts)} messages (latest ts={max_ts})")
    else:
        print("No non-empty messages to ingest.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Full re-index (ignore sync state)")
    args = parser.parse_args()
    builder(full_reindex=args.full)
    print("Done.")