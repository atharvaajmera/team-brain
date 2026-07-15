import json
import logging
import os
import ssl
import time

import re

import certifi
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from memory.settings import settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ingest")

SLACK_SYNC_TOKEN = settings.SLACK_SYNC_TOKEN

ssl_context = ssl.create_default_context(cafile=certifi.where())

client = WebClient(token=SLACK_SYNC_TOKEN, ssl=ssl_context)

# Always resolve the *bot app* user ID via the bot token.
# Sync token may be a user token (xoxp-); auth_test on that would return a human
# user_id and we'd incorrectly drop that person's messages from the archive.
_BOT_USER_ID = None
try:
    bot_token = settings.SLACK_BOT_TOKEN or SLACK_SYNC_TOKEN
    _auth = WebClient(token=bot_token, ssl=ssl_context).auth_test()
    _BOT_USER_ID = _auth.get("user_id")
    logger.info(f"Bot user ID resolved: {_BOT_USER_ID}")
except Exception as e:
    logger.warning(f"Could not resolve bot user ID: {e}")


# Pattern to match Slack bot mentions like <@U12345ABC>
_BOT_MENTION_RE = re.compile(r"<@[UW][A-Z0-9]+>")


def _is_bot_mention_query(text):
    """Return True if this message is just a user tagging the bot with a question.

    These are queries TO the bot, not team knowledge, and they pollute
    search results by being semantically identical to actual user queries.
    """
    text = (text or "").strip()
    # If the message starts with a bot mention, it's a query to the bot
    return bool(_BOT_MENTION_RE.match(text))


def _is_system_message(text):
    text = (text or "").lower()
    return any(
        phrase in text
        for phrase in [
            "joined the channel",
            "left the channel",
            "has joined",
            "has left",
            "added to the channel",
            "removed from the channel",
        ]
    )


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
                name = (
                    profile.get("real_name")
                    or profile.get("display_name")
                    or user.get("name")
                )
                if uid and name:
                    norm_name = name.lower().replace(" ", "_")
                    user_map[uid] = {"norm": norm_name, "display": name}

            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
    except SlackApiError as e:
        if "missing_scope" in str(e):
            logger.warning(
                "[user_map] Bot token missing 'users:read' scope. Author names will use raw Slack IDs."
            )
            logger.warning(
                "           Add 'users:read' to your Slack app's Bot Token Scopes to enable name resolution."
            )
        else:
            logger.error(f"[user_map] Error fetching users: {e}")
    except Exception as e:
        logger.error(f"[user_map] Unexpected error: {e}")
    return user_map


def get_public_channels():
    """Fetch all channels the sync token can access.

    Prefers both public and private channels when the token/scopes allow it,
    and falls back to public channels only on missing-scope errors.
    """
    channel_types_to_try = ["public_channel,private_channel", "public_channel"]

    for channel_types in channel_types_to_try:
        channels = []
        cursor = None
        try:
            while True:
                kwargs = {
                    "types": channel_types,
                    "limit": 200,
                    "exclude_archived": True,
                }
                if cursor:
                    kwargs["cursor"] = cursor

                response = client.conversations_list(**kwargs)
                for ch in response.get("channels", []):
                    if ch.get("id"):
                        channels.append(
                            {"id": ch["id"], "name": ch.get("name", ch["id"])}
                        )

                cursor = response.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break
            return channels
        except SlackApiError as e:
            if "missing_scope" in str(e) and channel_types != "public_channel":
                logger.warning(
                    "[channels] Sync token missing private-channel scope; retrying with public channels only."
                )
                continue
            if "missing_scope" in str(e):
                logger.warning(
                    "[channels] Sync token missing channel discovery scope. Will fall back to SLACK_CHANNEL_ID_AUTO."
                )
                logger.warning(
                    "           Add Slack history/read scopes to your sync token for multi-channel ingestion."
                )
            else:
                logger.error(f"[channels] Error fetching channels: {e}")
            return []
        except Exception as e:
            logger.error(f"[channels] Unexpected error: {e}")
            return []

    return []


def _get_author(msg, user_map=None):
    """Return the human-readable author tuple: (normalized_name, display_name, user_id)."""
    user_id = msg.get("user", "")

    try:
        # Check generator metadata first
        gen_name = msg["metadata"]["event_payload"]["author"]
        norm_name = gen_name.lower().replace(" ", "_")
        return (norm_name, gen_name, user_id)
    except (KeyError, TypeError):
        pass

    if not user_id:
        return ("unknown", "Unknown", "")

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
                    delay = int(e.response.headers.get("Retry-After", 10))
                    logger.warning(
                        f"Rate limited on history. Sleeping for {delay} seconds..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    raise e
            messages = [
                msg
                for msg in response.get("messages", [])
                if not _is_system_message(msg.get("text", ""))
                and msg.get("user") != _BOT_USER_ID
                and msg.get("subtype") != "bot_message"
                and not _is_bot_mention_query(msg.get("text", ""))
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
                                "limit": 200,
                            }
                            if reply_cursor:
                                reply_kwargs["cursor"] = reply_cursor

                            replies_response = client.conversations_replies(
                                **reply_kwargs
                            )
                            msgs_in_reply = replies_response.get("messages", [])
                            if not reply_cursor:
                                msgs_in_reply = msgs_in_reply[
                                    1:
                                ]  # Skip first as it's the root msg

                            replies = [
                                r
                                for r in msgs_in_reply
                                if not _is_system_message(r.get("text", ""))
                                and r.get("user") != _BOT_USER_ID
                                and r.get("subtype") != "bot_message"
                                and not _is_bot_mention_query(r.get("text", ""))
                            ]
                            for reply in replies:
                                reply["author"] = _get_author(reply, user_map)
                                message_array.append(reply)

                            reply_cursor = replies_response.get(
                                "response_metadata", {}
                            ).get("next_cursor")
                            if not reply_cursor:
                                break
                        except SlackApiError as e:
                            if e.response.status_code == 429:
                                delay = int(e.response.headers.get("Retry-After", 10))
                                logger.warning(
                                    f"Rate limited on replies. Sleeping for {delay} seconds..."
                                )
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
