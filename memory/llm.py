import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("MODEL", "llama3.2")

def _build_context(threads):
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
    category_instructions = {
        "NARROW": (
            "The retrieval system found ONE highly relevant thread. "
            "Answer the user's question using the thread below. Be specific and concise."
        ),
        "AMBIGUOUS": (
            "The retrieval system found MULTIPLE possibly relevant threads. "
            "Summarise what each thread covers and ask the user to clarify which one they mean, "
            "or answer broadly if the threads are related."
        ),
        "BROAD": (
            "The query is broad and touches multiple threads. "
            "Provide a summary across all relevant threads. Highlight key themes."
        ),
        "REJECT": (
            "No relevant threads were found for this query. "
            "Politely let the user know and suggest they refine their question."
        ),
    }

    instruction = category_instructions.get(category, category_instructions["REJECT"])

    return (
        f"You are a helpful assistant for a software engineering team. "
        f"You answer questions based on archived Slack conversations.\n\n"
        f"Category: {category}\n"
        f"Instruction: {instruction}\n\n"
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
    context = _build_context(threads)
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
