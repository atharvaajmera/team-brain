import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
import sys
from ingest import get_threads_from_channel, get_user_map, get_public_channels
from memory.storage import add_messages, reset_collection
from memory.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("brain")

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
    if full_reindex:
        logger.info("Full sync: wiping existing collection and fetching all messages")
        reset_collection()
        if SYNC_STATE_FILE.exists():
            SYNC_STATE_FILE.unlink()
        last_ts = None
    else:
        last_ts = _load_last_ts()
        if last_ts:
            logger.info(f"Incremental sync: fetching messages after ts={last_ts}")
        else:
            logger.info("Full sync: fetching all messages")

    # --- User mapping (graceful: empty map means raw Slack IDs) ---
    logger.info("Fetching user mapping...")
    user_map = get_user_map()
    if user_map:
        logger.info(f"Loaded {len(user_map)} users.")
    else:
        logger.info("User mapping unavailable. Authors will use raw Slack IDs or generator names.")

    # --- Channel discovery (graceful: falls back to SLACK_CHANNEL_ID_AUTO) ---
    logger.info("Discovering channels...")
    channel_list = get_public_channels()

    if channel_list:
        logger.info(f"Found {len(channel_list)} public channels.")
        channels = [(ch["id"], ch["name"]) for ch in channel_list]
    else:
        fallback_id = settings.SLACK_CHANNEL_ID_AUTO
        if not fallback_id:
            logger.error("No channels discovered and SLACK_CHANNEL_ID_AUTO is not set.")
            logger.error("Either add 'channels:read' scope or set SLACK_CHANNEL_ID_AUTO in .env")
            sys.exit(1)
        logger.info(f"Falling back to single channel: {fallback_id}")
        channels = [(fallback_id, fallback_id)]

    texts, ids, metadatas = [], [], []
    max_ts = 0

    for channel_id, channel_name in channels:
        logger.info(f"Ingesting #{channel_name} ({channel_id})...")
        messages = get_threads_from_channel(channel_id, user_map=user_map, after_ts=last_ts)
        if not messages:
            logger.info(f"  No new messages in #{channel_name}.")
            continue

        count = 0
        for msg in messages:
            text = msg.get("text", "")
            if not text.strip():
                continue
            author_val = msg.get("author", ("unknown", "Unknown", msg.get("user", "")))
            if isinstance(author_val, tuple) and len(author_val) == 3:
                norm_name, display_name, author_id = author_val
            else:
                norm_name = display_name = author_id = str(author_val)
                
            ts = float(msg.get("ts", 0))
            thread_id = float(msg.get("thread_ts", ts))
            max_ts = max(max_ts, ts)

            text_to_embed = f"{display_name}: {text}"

            texts.append(text_to_embed)
            ids.append(f"slack:{channel_id}:{ts}")

            metadatas.append({
                "author": norm_name,
                "author_id": author_id,
                "author_display": display_name,
                "ts": ts,
                "text": text,
                "thread_id": thread_id,
                "channel_id": channel_id,
            })
            count += 1

        logger.info(f"  Collected {count} messages from #{channel_name}.")

    if texts:
        add_messages(texts, ids, metadatas)
        _save_last_ts(max_ts)
        logger.info(f"Ingested {len(texts)} total messages across {len(channels)} channel(s) (latest ts={max_ts})")
    else:
        logger.info("No non-empty messages to ingest.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Full re-index (ignore sync state)")
    args = parser.parse_args()
    builder(full_reindex=args.full)
    logger.info("Done.")