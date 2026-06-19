import os
import ssl
import certifi
import json
import time
import logging
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from memory.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingest")

SLACK_BOT_TOKEN = settings.SLACK_BOT_TOKEN

ssl_context = ssl.create_default_context(cafile=certifi.where())

client= WebClient(token=SLACK_BOT_TOKEN,ssl=ssl_context)


def _is_system_message(text):
    text = (text or "").lower()
    return any(phrase in text for phrase in [
        "joined the channel",
        "left the channel",
        "has joined",
        "has left",
        "added to the channel",
        "removed from the channel",
    ])

def get_user_map():
    """Fetch all Slack users and return a dict of {id: real_name or name}.
    
    Requires the `users:read` scope on the bot token.
    Returns an empty dict if the scope is missing (graceful degradation).
    """
    user_map = {}
    cursor = None
    try:
        while True:
            kwargs = {"limit": 200}
            if cursor:
                kwargs["cursor"] = cursor
            
            response = client.users_list(**kwargs)
            for user in response.get("members", []):
                uid = user.get("id")
                # Prefer real_name, fallback to display_name, then name
                profile = user.get("profile", {})
                name = profile.get("real_name") or profile.get("display_name") or user.get("name")
                if uid and name:
                    norm_name = name.lower().replace(" ", "_")
                    user_map[uid] = {"norm": norm_name, "display": name}
                    
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
    except SlackApiError as e:
        if "missing_scope" in str(e):
            logger.warning("[user_map] Bot token missing 'users:read' scope. Author names will use raw Slack IDs.")
            logger.warning("           Add 'users:read' to your Slack app's Bot Token Scopes to enable name resolution.")
        else:
            logger.error(f"[user_map] Error fetching users: {e}")
    except Exception as e:
        logger.error(f"[user_map] Unexpected error: {e}")
    return user_map


def get_public_channels():
    """Fetch all public channels the bot can see.
    
    Requires the `channels:read` scope on the bot token.
    Returns an empty list if the scope is missing (graceful degradation).
    """
    channels = []
    cursor = None
    try:
        while True:
            kwargs = {"types": "public_channel", "limit": 200, "exclude_archived": True}
            if cursor:
                kwargs["cursor"] = cursor
            
            response = client.conversations_list(**kwargs)
            for ch in response.get("channels", []):
                if ch.get("id"):
                    channels.append({"id": ch["id"], "name": ch.get("name", ch["id"])})
                    
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
    except SlackApiError as e:
        if "missing_scope" in str(e):
            logger.warning("[channels] Bot token missing 'channels:read' scope. Will fall back to SLACK_CHANNEL_ID_AUTO.")
            logger.warning("           Add 'channels:read' to your Slack app's Bot Token Scopes for multi-channel ingestion.")
        else:
            logger.error(f"[channels] Error fetching channels: {e}")
    except Exception as e:
        logger.error(f"[channels] Unexpected error: {e}")
    return channels

def _get_author(msg, user_map=None):
    """Return the human-readable author tuple: (normalized_name, display_name, user_id)."""
    user_id = msg.get('user', '')
    
    try:
        # Check generator metadata first
        gen_name = msg['metadata']['event_payload']['author']
        norm_name = gen_name.lower().replace(" ", "_")
        return (norm_name, gen_name, user_id)
    except (KeyError, TypeError):
        pass
    
    if not user_id:
        return ('unknown', 'Unknown', '')
    
    if user_map and user_id in user_map:
        info = user_map[user_id]
        return (info["norm"], info["display"], user_id)
        
    return (user_id, user_id, user_id)


def get_threads_from_channel(channel_id, user_map=None, limit=None, after_ts=None):
    """Fetch messages with cursor-based pagination.

    Args:
        channel_id: Slack channel ID
        user_map: mapping from user id to name
        limit: max total messages (None = all)
        after_ts: Unix timestamp for incremental sync (fetch only newer messages)
    """
    message_array = []
    cursor = None
    batch_size = 200

    try:
        while True:
            kwargs = {
                "channel": channel_id,
                "limit": batch_size,
                "include_all_metadata": True,
            }
            if cursor:
                kwargs["cursor"] = cursor
            if after_ts:
                kwargs["oldest"] = str(after_ts)

            try:
                response = client.conversations_history(**kwargs)
            except SlackApiError as e:
                if e.response.status_code == 429:
                    delay = int(e.response.headers.get('Retry-After', 10))
                    logger.warning(f"Rate limited on history. Sleeping for {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    raise e
            messages = [
                msg for msg in response.get("messages", [])
                if not _is_system_message(msg.get("text", ""))
            ]

            for msg in messages:
                author = _get_author(msg, user_map)
                msg["author"] = author
                message_array.append(msg)

                if msg.get("reply_count", 0) > 0:
                    thread_ts = msg.get("ts")
                    reply_cursor = None
                    while True:
                        try:
                            reply_kwargs = {
                                "channel": channel_id,
                                "ts": thread_ts,
                                "include_all_metadata": True,
                                "limit": 200
                            }
                            if reply_cursor:
                                reply_kwargs["cursor"] = reply_cursor
                            
                            replies_response = client.conversations_replies(**reply_kwargs)
                            msgs_in_reply = replies_response.get("messages", [])
                            if not reply_cursor:
                                msgs_in_reply = msgs_in_reply[1:] # Skip first as it's the root msg
                            
                            replies = [
                                r for r in msgs_in_reply
                                if not _is_system_message(r.get("text", ""))
                            ]
                            for reply in replies:
                                reply["author"] = _get_author(reply, user_map)
                                message_array.append(reply)
                                
                            reply_cursor = replies_response.get("response_metadata", {}).get("next_cursor")
                            if not reply_cursor:
                                break
                        except SlackApiError as e:
                            if e.response.status_code == 429:
                                delay = int(e.response.headers.get('Retry-After', 10))
                                logger.warning(f"Rate limited on replies. Sleeping for {delay} seconds...")
                                time.sleep(delay)
                            else:
                                logger.error(f"Error fetching replies: {e}")
                                break

                if limit and len(message_array) >= limit:
                    return message_array[:limit]

            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        return message_array
    except Exception as e:
        logger.error(f"Error fetching messages: {e}")
        return message_array


if __name__ == "__main__":
    CHANNEL_ID = settings.SLACK_CHANNEL_ID
    if not CHANNEL_ID:
        raise ValueError("SLACK_CHANNEL_ID environment variable is not set")
    msgs = get_threads_from_channel(CHANNEL_ID, limit=10)
    logger.info(f"Fetched {len(msgs)} messages")
    for m in msgs[:5]:
        logger.info(f"  @{m.get('author')}: {m.get('text', '')[:80]}")
