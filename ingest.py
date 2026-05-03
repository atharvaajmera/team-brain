import os
import ssl
import certifi
import json
from slack_sdk import WebClient
from dotenv import load_dotenv

load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

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


def get_threads_from_channel(channel_id, limit=10):
    message_array = []
    try:
        response=client.conversations_history(
            channel=channel_id,
            limit=limit,
            include_all_metadata=True,
        )
        messages = [
            msg for msg in response['messages']
            if not _is_system_message(msg.get('text', ''))
        ]
        for msg in messages:
            author = _get_author(msg)
            print(f"User: {author}, Text: {msg.get('text')}")
            msg['author'] = author
            message_array.append(msg)
            if(msg.get('reply_count', 0) > 0):
                thread_ts = msg.get('ts')
                replies_response = client.conversations_replies(
                    channel=channel_id,
                    ts=thread_ts,
                    include_all_metadata=True,
                )
                replies = [
                    reply for reply in replies_response['messages'][1:]
                    if not _is_system_message(reply.get('text', ''))
                ]
                for reply in replies:
                    reply_author = _get_author(reply)
                    print(f"  Reply from User: {reply_author}, Text: {reply.get('text')}")
                    reply['author'] = reply_author
                    message_array.append(reply)
        return message_array
    except Exception as e:
        print(f"Error fetching messages: {e}")
        return []

if __name__ == "__main__":
    CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID")
    if not CHANNEL_ID:
        raise ValueError("SLACK_CHANNEL_ID environment variable is not set")
    get_threads_from_channel(CHANNEL_ID)
