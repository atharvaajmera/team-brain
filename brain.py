import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from ingest import get_public_channels, get_threads_from_channel, get_user_map
from memory.settings import settings
from memory.storage import add_messages, reset_collection

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("brain")

REPO_ROOT = Path(__file__).resolve().parent
SYNC_STATE_FILE = REPO_ROOT / "config" / "sync_state.json"
SYNC_OVERLAP_SECONDS = 7 * 24 * 60 * 60


def _load_channel_ts(channel_id):
    if SYNC_STATE_FILE.exists():
        try:
            with open(SYNC_STATE_FILE) as f:
                data = json.load(f)
                if "last_ts" in data:
                    data = {"channels": {}}
                return data.get("channels", {}).get(channel_id, {}).get("last_ts")
        except Exception:
            pass
    return None


def _save_channel_ts(channel_id, ts):
    SYNC_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    data = {"channels": {}}
    if SYNC_STATE_FILE.exists():
        try:
            with open(SYNC_STATE_FILE) as f:
                old_data = json.load(f)
                if "last_ts" not in old_data:
                    data = old_data
        except Exception:
            pass

    data.setdefault("channels", {})
    data["channels"].setdefault(channel_id, {})
    data["channels"][channel_id]["last_ts"] = ts

    with open(SYNC_STATE_FILE, "w") as f:
        json.dump(data, f)


def _with_sync_overlap(last_ts):
    if last_ts is None:
        return None
    try:
        return max(float(last_ts) - SYNC_OVERLAP_SECONDS, 0)
    except (TypeError, ValueError):
        return None


def builder(full_reindex=False):
    if full_reindex:
        logger.info("Full sync: wiping existing collection and fetching all messages")
        reset_collection()
        if SYNC_STATE_FILE.exists():
            SYNC_STATE_FILE.unlink()
    else:
        logger.info("Incremental sync: fetching messages")

    # --- User mapping (graceful: empty map means raw Slack IDs) ---
    logger.info("Fetching user mapping...")
    user_map = get_user_map()
    if user_map:
        logger.info(f"Loaded {len(user_map)} users.")
    else:
        logger.info(
            "User mapping unavailable. Authors will use raw Slack IDs or generator names."
        )

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
            logger.error(
                "Either add 'channels:read' scope or set SLACK_CHANNEL_ID_AUTO in .env"
            )
            sys.exit(1)
        logger.info(f"Falling back to single channel: {fallback_id}")
        channels = [(fallback_id, fallback_id)]

    texts, ids, metadatas = [], [], []
    channel_max_ts = {}

    for channel_id, channel_name in channels:
        logger.info(f"Ingesting #{channel_name} ({channel_id})...")
        last_ts = _load_channel_ts(channel_id) if not full_reindex else None
        fetch_after_ts = _with_sync_overlap(last_ts)
        if last_ts and fetch_after_ts is not None and fetch_after_ts < float(last_ts):
            logger.info(
                "  Re-fetching a 7-day overlap window to catch late thread replies."
            )

        messages = get_threads_from_channel(
            channel_id, user_map=user_map, after_ts=fetch_after_ts
        )
        if not messages:
            logger.info(f"  No new messages in #{channel_name}.")
            continue

        count = 0
        max_ts = 0
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

            metadatas.append(
                {
                    "author": norm_name,
                    "author_id": author_id,
                    "author_display": display_name,
                    "ts": ts,
                    "text": text,
                    "thread_id": thread_id,
                    "channel_id": channel_id,
                }
            )
            count += 1

        logger.info(f"  Collected {count} messages from #{channel_name}.")
        if max_ts > 0:
            channel_max_ts[channel_id] = max_ts

    if texts:
        add_messages(texts, ids, metadatas)
        for cid, ts in channel_max_ts.items():
            _save_channel_ts(cid, ts)
        logger.info(
            f"Ingested {len(texts)} total messages across {len(channels)} channel(s)"
        )
    else:
        logger.info("No non-empty messages to ingest.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true", help="Full re-index (ignore sync state)"
    )
    args = parser.parse_args()
    builder(full_reindex=args.full)
    logger.info("Done.")
