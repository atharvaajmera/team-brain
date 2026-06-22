import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from memory.service import answer_query, is_corpus_ready
from memory.settings import settings
from memory.slack_access import _get_allowed_channels

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("slack_bot")

SLACK_BOT_TOKEN = settings.SLACK_BOT_TOKEN
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN", "")
MAX_WORKERS = max(1, int(os.getenv("SLACK_WORKER_THREADS", "4")))
MAX_PENDING_REQUESTS = max(MAX_WORKERS, int(os.getenv("SLACK_MAX_PENDING", "16")))

if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
    logger.error(
        "Both SLACK_BOT_TOKEN and SLACK_APP_TOKEN are required to run the bot in Socket Mode."
    )
    raise SystemExit(1)

_executor = ThreadPoolExecutor(
    max_workers=MAX_WORKERS, thread_name_prefix="slack-query"
)
_pending_requests = threading.BoundedSemaphore(MAX_PENDING_REQUESTS)
app = App(token=SLACK_BOT_TOKEN)


def _strip_bot_mention(text: str) -> str:
    """Remove the bot mention from the message text."""
    return re.sub(r"<@[UW][A-Z0-9]+>", "", text).strip()


def _format_slack_response(result) -> str:
    """Format the QueryResponse into a Slack-friendly string."""
    if result.status == "reject":
        return f"❌ {result.answer}"
    if result.status == "clarify":
        return f"❓ {result.clarification_question}"
    if result.status == "error":
        return f"⚠️ Something went wrong: {result.answer}"

    msg = result.answer
    if result.citations:
        msg += "\n\n*Sources:*\n"
        for i, c in enumerate(result.citations, 1):
            link = f"<{c.permalink}|{c.readable_ts}>" if c.permalink else c.readable_ts
            msg += f"[{i}] {c.author} at {link}\n"
    return msg


def _safe_update_or_reply(
    client, channel_id: str, ack_ts: str | None, thread_ts: str, text: str
):
    if ack_ts:
        client.chat_update(channel=channel_id, ts=ack_ts, text=text)
    else:
        client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=text)


@app.event("app_mention")
def handle_mention(event, say, client):
    if event.get("bot_id"):
        return

    user_id = event.get("user")
    if not user_id:
        return

    channel_id = event["channel"]
    thread_ts = event.get("thread_ts") or event["ts"]
    text = _strip_bot_mention(event.get("text", ""))

    logger.info("Received mention from %s in %s: %s", user_id, channel_id, text)
    allowed_channels = _get_allowed_channels(client, user_id, channel_id)
    acquired_slot = False

    try:
        if not _pending_requests.acquire(blocking=False):
            say(
                text="⚠️ I’m handling too many requests right now. Please try again in a minute.",
                thread_ts=thread_ts,
            )
            return
        acquired_slot = True

        ack_res = say(text="🔍 Looking into that...", thread_ts=thread_ts)
        ack_ts = ack_res.get("ts") if isinstance(ack_res, dict) else None

        def process_query():
            try:
                result = answer_query(
                    query=text,
                    source="slack",
                    user_id=user_id,
                    channel_id=channel_id,
                    allowed_channel_ids=allowed_channels,
                )
                reply_text = _format_slack_response(result)
                if len(reply_text) > 35000:
                    reply_text = reply_text[:34997] + "..."
                _safe_update_or_reply(client, channel_id, ack_ts, thread_ts, reply_text)
                logger.info("Successfully responded to %s in %s", user_id, channel_id)
            except Exception as e:
                logger.error("Error handling mention in thread: %s", e, exc_info=True)
                _safe_update_or_reply(
                    client,
                    channel_id,
                    ack_ts,
                    thread_ts,
                    "⚠️ Sorry, I encountered an internal error while processing your request.",
                )
            finally:
                _pending_requests.release()

        _executor.submit(process_query)
        acquired_slot = False

    except Exception as e:
        if acquired_slot:
            _pending_requests.release()
        logger.error("Error initiating mention handling: %s", e, exc_info=True)
        say(
            text="⚠️ Sorry, I encountered an internal error starting your request.",
            thread_ts=thread_ts,
        )


class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        elif self.path == "/readyz":
            if SLACK_BOT_TOKEN and SLACK_APP_TOKEN and is_corpus_ready():
                self.send_response(200)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Ready")
            else:
                self.send_response(503)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Not Ready")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def start_health_server(port=8080):
    server = HTTPServer(("", port), HealthCheckHandler)
    logger.info("Starting health check server on port %s", port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()


if __name__ == "__main__":
    logger.info("Starting Slack bot in Socket Mode...")
    start_health_server(port=int(os.getenv("PORT", "8080")))
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
