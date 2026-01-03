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

def get_threads_from_channel(channel_id, limit=10):
    message_array = []
    try:
        response=client.conversations_history(
            channel=channel_id,
            limit=limit
        )
        messages=response['messages']
        for msg in messages:
            print(f"User: {msg.get('user')}, Text: {msg.get('text')}")
            message_array.append(msg)
            if(msg.get('reply_count', 0) > 0):
                thread_ts = msg.get('ts')
                replies_response = client.conversations_replies(
                    channel=channel_id,
                    ts=thread_ts
                )
                replies = replies_response['messages'][1:]  
                for reply in replies:
                    print(f"  Reply from User: {reply.get('user')}, Text: {reply.get('text')}")
                    message_array.append(reply)
        return message_array
    except Exception as e:
        print(f"Error fetching messages: {e}")
        return []

if __name__ == "__main__":
    CHANNEL_ID="C0A4LV0HZJ7"
    get_threads_from_channel(CHANNEL_ID)