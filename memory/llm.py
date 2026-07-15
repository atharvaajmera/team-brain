import json

import requests

from memory.settings import settings

OLLAMA_URL = settings.OLLAMA_URL
MODEL = settings.MODEL


# Headers to bypass ngrok's free-tier browser interstitial
_NGROK_HEADERS = {"ngrok-skip-browser-warning": "true"}


def is_ollama_available(timeout=2):
    """Quick health check: can we reach Ollama?"""
    try:
        health_url = OLLAMA_URL.replace("/api/generate", "")
        resp = requests.get(health_url, timeout=timeout, headers=_NGROK_HEADERS)
        return resp.status_code == 200
    except Exception:
        return False


def _message_body(msg: dict) -> str:
    """Prefer raw message text; strip leading 'Author: ' prefix from embedded docs."""
    meta = msg.get("metadata") or {}
    text = (meta.get("text") or msg.get("document") or "").strip()
    if not text:
        return ""
    # Stored docs are often "Display Name: actual message"
    author = (
        meta.get("author_display")
        or meta.get("author")
        or meta.get("user")
        or ""
    )
    if author:
        prefix = f"{author}:"
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].lstrip()
        norm = str(author).lower().replace(" ", "_")
        if norm and text.lower().startswith(f"{norm}:"):
            text = text[len(norm) + 1 :].lstrip()
    return text


def build_context(threads, include_permalinks=True):
    if not threads:
        return "(no relevant threads found)"

    from memory.citations import make_permalink, ts_to_readable

    parts = []
    for i, thread in enumerate(threads, 1):
        msgs = thread.get("messages", [])
        lines = []
        for msg in msgs:
            meta = msg.get("metadata") or {}
            user = meta.get("author_display") or meta.get("author") or meta.get("user") or "unknown"
            ts = meta.get("ts", "")
            text = _message_body(msg)
            readable_ts = ts_to_readable(ts)
            channel_id = meta.get("channel_id", "")
            permalink = make_permalink(channel_id, ts) if include_permalinks else ""

            if permalink:
                lines.append(f"  @{user} [{readable_ts}] ({permalink}): {text}")
            else:
                lines.append(f"  @{user} [{readable_ts}]: {text}")

        thread_id = thread.get("thread_id", "?")
        readable_thread_ts = ts_to_readable(thread_id)
        parts.append(f"Thread {i} (started {readable_thread_ts}):\n" + "\n".join(lines))

    return "\n\n".join(parts)


def _build_prompt(query, category, context, answer_reqs=None):
    """Build an LLM prompt for local answer generation."""
    answer_reqs = answer_reqs or {}
    format_str = answer_reqs.get("format", "direct")
    cite = answer_reqs.get("cite_sources", True)

    cite_rule = (
        (
            "- When citing, reference the author and timestamp only: e.g. '@alice (2026-05-03 14:30)'.\n"
            "- Do NOT invent permalinks or write 'Permalink: Not available'.\n"
            "- Do NOT paste raw Slack URLs; the system appends source links separately.\n"
            "- Quote message body only — do not repeat the author name inside the quote."
        )
        if cite
        else "- No need for explicit citations."
    )
    format_rule = (
        f"- Format your response as a {format_str}. Keep it concise (2-4 sentences for direct answers)."
    )

    return (
        "You are a helpful assistant for a software engineering team. "
        "You answer questions based on archived Slack conversations.\n\n"
        "Important rules:\n"
        "- Base your answer ONLY on the provided Slack threads.\n"
        "- If the threads are not relevant, say so clearly.\n"
        "- Do not fabricate information.\n"
        f"{cite_rule}\n"
        f"{format_rule}\n\n"
        f"--- Retrieved Slack threads ---\n{context}\n"
        f"--- End of threads ---\n\n"
        f"User question: {query}\n\n"
        f"Answer:"
    )


def _stream_response(payload):
    """Yield tokens one by one from Ollama streaming endpoint."""
    resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120, headers=_NGROK_HEADERS)
    resp.raise_for_status()
    for line in resp.iter_lines():
        if line:
            chunk = json.loads(line)
            token = chunk.get("response", "")
            if token:
                yield token
            if chunk.get("done"):
                break


def generate_response(query, category, threads, stream=False, answer_reqs=None):
    context = build_context(threads)
    prompt = _build_prompt(query, category, context, answer_reqs)

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Local Route Triggered: Sending query to Ollama at %s", OLLAMA_URL)
    
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": stream,
    }

    if stream:
        return _stream_response(payload)

    resp = requests.post(OLLAMA_URL, json=payload, timeout=120, headers=_NGROK_HEADERS)
    resp.raise_for_status()
    return resp.json().get("response", "")
