import os
import ssl
import certifi
import json
import time
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from memory.settings import settings

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


def _get_author(msg):
    """Return the human-readable author name.
    For messages posted by the generator, reads from metadata.event_payload.author.
    Falls back to the raw Slack user ID for real messages.
    """
    try:
        return msg['metadata']['event_payload']['author']
    except (KeyError, TypeError):
        return msg.get('user', 'unknown')


def get_threads_from_channel(channel_id, limit=None, after_ts=None):
    """Fetch messages with cursor-based pagination.

    Args:
        channel_id: Slack channel ID
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
                    print(f"Rate limited on history. Sleeping for {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    raise e
            messages = [
                msg for msg in response.get("messages", [])
                if not _is_system_message(msg.get("text", ""))
            ]

            for msg in messages:
                author = _get_author(msg)
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
                                reply["author"] = _get_author(reply)
                                message_array.append(reply)
                                
                            reply_cursor = replies_response.get("response_metadata", {}).get("next_cursor")
                            if not reply_cursor:
                                break
                        except SlackApiError as e:
                            if e.response.status_code == 429:
                                delay = int(e.response.headers.get('Retry-After', 10))
                                print(f"Rate limited on replies. Sleeping for {delay} seconds...")
                                time.sleep(delay)
                            else:
                                print(f"Error fetching replies: {e}")
                                break

                if limit and len(message_array) >= limit:
                    return message_array[:limit]

            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        return message_array
    except Exception as e:
        print(f"Error fetching messages: {e}")
        return message_array


if __name__ == "__main__":
    CHANNEL_ID = settings.SLACK_CHANNEL_ID
    if not CHANNEL_ID:
        raise ValueError("SLACK_CHANNEL_ID environment variable is not set")
    msgs = get_threads_from_channel(CHANNEL_ID, limit=10)
    print(f"Fetched {len(msgs)} messages")
    for m in msgs[:5]:
        print(f"  @{m.get('author')}: {m.get('text', '')[:80]}")
