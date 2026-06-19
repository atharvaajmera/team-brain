import os
import re
import logging
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from memory.settings import settings
from memory.service import answer_query

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("slack_bot")

SLACK_BOT_TOKEN = settings.SLACK_BOT_TOKEN
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN", "")

if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
    logger.error("Both SLACK_BOT_TOKEN and SLACK_APP_TOKEN are required to run the bot in Socket Mode.")
    import sys
    sys.exit(1)

app = App(token=SLACK_BOT_TOKEN)

def _strip_bot_mention(text: str) -> str:
    """Removes the bot mention from the start of the message."""
    # Matches <@U123456> at the beginning or anywhere in the text
    return re.sub(r'<@[UW][A-Z0-9]+>', '', text).strip()

def _format_slack_response(result) -> str:
    """Format the QueryResponse into a Slack-friendly string."""
    if result.status == "reject":
        return f"❌ {result.answer}"
    elif result.status == "clarify":
        return f"❓ {result.clarification_question}"
    elif result.status == "error":
        return f"⚠️ Something went wrong: {result.answer}"
    
    # Success response
    msg = result.answer
    if result.citations:
        msg += "\n\n*Sources:*\n"
        for i, c in enumerate(result.citations):
            link = f"<{c.permalink}|{c.readable_ts}>" if c.permalink else c.readable_ts
            msg += f"[{i+1}] {c.author} at {link}\n"
    
    return msg

import time
_user_channel_cache = {}

def _get_allowed_channels(client, user_id):
    """Fetch and cache the list of channels a user has access to."""
    now = time.time()
    cached, ts = _user_channel_cache.get(user_id, (None, 0))
    if cached and now - ts < 300:
        return cached

    allowed = []
    try:
        cursor = None
        while True:
            response = client.users_conversations(
                user=user_id,
                exclude_archived=True,
                types="public_channel,private_channel",
                cursor=cursor
            )
            for channel in response.get("channels", []):
                allowed.append(channel["id"])
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
        _user_channel_cache[user_id] = (allowed, now)
        return allowed
    except Exception as e:
        logger.warning(f"Could not fetch allowed channels for {user_id} (missing scope?): {e}")
        return None  # Opt-in access control: if we can't fetch, we don't restrict

@app.event("app_mention")
def handle_mention(event, say, client):
    if event.get("bot_id"):
        return

    user_id = event.get("user")
    if not user_id:
        return

    channel_id = event["channel"]
    thread_ts = event.get("thread_ts") or event["ts"]
    text = _strip_bot_mention(event["text"])
    
    logger.info(f"Received mention from {user_id} in {channel_id}: {text}")
    
    allowed_channels = _get_allowed_channels(client, user_id)

    try:
        result = answer_query(
            query=text,
            source="slack",
            user_id=user_id,
            channel_id=channel_id,
            allowed_channel_ids=allowed_channels,
        )
        
        reply_text = _format_slack_response(result)
        
        # Slack blocks have a 3000 character limit per text block. 
        # But we're just using basic markdown text here.
        # Max message length is 40,000 characters. We'll truncate to 35,000 just in case.
        if len(reply_text) > 35000:
            reply_text = reply_text[:34997] + "..."
            
        say(text=reply_text, thread_ts=thread_ts)
        logger.info(f"Successfully responded to {user_id} in {channel_id}")
    except Exception as e:
        logger.error(f"Error handling mention: {e}", exc_info=True)
        say(text="⚠️ Sorry, I encountered an internal error while processing your request.", thread_ts=thread_ts)

import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        elif self.path == "/readyz":
            # Just verify we have tokens and can start
            if SLACK_BOT_TOKEN and SLACK_APP_TOKEN:
                self.send_response(200)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Ready")
            else:
                self.send_response(503)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()
            
    def log_message(self, format, *args):
        # Suppress HTTP logging to avoid spam
        pass

def start_health_server(port=8080):
    server = HTTPServer(("", port), HealthCheckHandler)
    logger.info(f"Starting health check server on port {port}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

if __name__ == "__main__":
    logger.info("Starting Slack bot in Socket Mode...")
    start_health_server(port=8080)
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
