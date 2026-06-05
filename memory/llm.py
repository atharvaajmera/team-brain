import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("MODEL", "llama3.2")

def is_ollama_available(timeout=2):
    """Quick health check: can we reach Ollama?"""
    try:
        health_url = OLLAMA_URL.replace("/api/generate", "")
        resp = requests.get(health_url, timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def build_context(threads):
    if not threads:
        return "(no relevant threads found)"

    parts = []
    for i, thread in enumerate(threads, 1):
        msgs = thread.get('messages', [])
        lines = []
        for msg in msgs:
            meta = msg.get('metadata', {})
            user = meta.get('author', meta.get('user', 'unknown'))
            ts = meta.get('ts', '')
            text = msg.get('document', '')
            lines.append(f"  @{user} [{ts}]: {text}")
        parts.append(f"Thread {i} (id: {thread.get('thread_id', '?')}):\n" + "\n".join(lines))

    return "\n\n".join(parts)


def _build_prompt(query, category, context):
    """Build an LLM prompt for local answer generation.

    The category parameter is accepted for backward compatibility but is no
    longer used — the LLM adapts its response style based on the retrieved
    context itself.
    """
    return (
        "You are a helpful assistant for a software engineering team. "
        "You answer questions based on archived Slack conversations.\n\n"
        "Important rules:\n"
        "- Base your answer ONLY on the provided Slack threads.\n"
        "- If one thread clearly answers the question, be specific and concise.\n"
        "- If multiple threads are relevant, summarize across them.\n"
        "- If the threads are not relevant, say so clearly.\n"
        "- Do not fabricate information.\n"
        "- Reference specific authors and timestamps when useful.\n\n"
        f"--- Retrieved Slack threads ---\n{context}\n"
        f"--- End of threads ---\n\n"
        f"User question: {query}\n\n"
        f"Answer:"
    )


def _stream_response(payload):
    """Yield tokens one by one from Ollama streaming endpoint."""
    resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)
    resp.raise_for_status()
    for line in resp.iter_lines():
        if line:
            chunk = json.loads(line)
            token = chunk.get("response", "")
            if token:
                yield token
            if chunk.get("done"):
                break


def generate_response(query, category, threads, stream=False):
    context = build_context(threads)
    prompt = _build_prompt(query, category, context)

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": stream,
    }

    if stream:
        return _stream_response(payload)

    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json().get("response", "")
